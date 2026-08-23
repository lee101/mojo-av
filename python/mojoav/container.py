from __future__ import annotations

import os
import struct
from fractions import Fraction
from pathlib import Path

import numpy as np

from ._lib import addr, lib
from .packet import Packet
from .stream import CodecContext, Stream, StreamContainer

AV_TIME_BASE = 1_000_000


class ContainerFormat:
    def __init__(self, name: str, long_name: str):
        self.name = name
        self.long_name = long_name

    def __repr__(self) -> str:
        return f"<mojoav.ContainerFormat {self.name}>"


class InputContainer:
    def __init__(self, file, *, format: str | None = None, **kwargs):
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported open option(s): {names}")
        self._closed = False
        if hasattr(file, "read"):
            self.name = getattr(file, "name", str(file))
            data = file.read()
        else:
            self.name = os.fspath(file)
            with builtins_open(os.fspath(file), "rb") as handle:
                data = handle.read()
        self._data = bytes(data)
        self.size = len(self._data)
        self.metadata: dict[str, str] = {}
        detected = _detect_format(self._data, format)
        if detected == "wav":
            self._init_wav()
        elif detected == "ivf":
            self._init_ivf()
        else:
            raise ValueError(f"unsupported container format: {detected}")

    def _init_wav(self) -> None:
        info = _parse_wav(self._data)
        self.format = ContainerFormat("wav", "WAV / WAVE (Waveform Audio)")
        codec = CodecContext(
            info["codec"],
            sample_rate=info["sample_rate"],
            channels=info["channels"],
            format=info["sample_format"],
        )
        duration = info["data_size"] // info["block_align"]
        self._stream = Stream(
            type="audio",
            time_base=Fraction(1, info["sample_rate"]),
            codec_context=codec,
            duration=duration,
        )
        self.streams = StreamContainer([self._stream])
        self.duration = (
            duration * AV_TIME_BASE + info["sample_rate"] // 2
        ) // info["sample_rate"]
        self.start_time = None
        self.bit_rate = (
            0 if not self.duration else self.size * 8 * AV_TIME_BASE // self.duration
        )
        capacity = (duration + 4095) // 4096
        offsets = np.empty(capacity, dtype=np.int64)
        sizes = np.empty(capacity, dtype=np.int64)
        durations = np.empty(capacity, dtype=np.int64)
        count = lib().mav_fixed_packets(
            info["data_size"],
            info["block_align"],
            4096,
            addr(offsets),
            addr(sizes),
            addr(durations),
            capacity,
        )
        if count < 0:
            raise ValueError("invalid WAV packet layout")
        packet_offsets = offsets[:count] + info["data_start"]
        packet_pts = np.concatenate(
            (np.array([0], dtype=np.int64), np.cumsum(durations[:count - 1]))
        ) if count else np.empty(0, dtype=np.int64)
        self._packet_offsets = memoryview(packet_offsets)
        self._packet_sizes = memoryview(sizes)[:count]
        self._packet_pts = memoryview(packet_pts)
        self._packet_durations = memoryview(durations)[:count]
        self._keyframes = None

    def _init_ivf(self) -> None:
        info = _parse_ivf_header(self._data)
        self.format = ContainerFormat("ivf", "On2 IVF")
        codec = CodecContext(
            "vp8", width=info["width"], height=info["height"], format="yuv420p"
        )
        time_base = Fraction(info["scale"], info["rate"])
        self._stream = Stream(
            type="video",
            time_base=time_base,
            codec_context=codec,
            duration=info["frames"],
            frames=info["frames"],
            start_time=0,
            base_rate=Fraction(info["rate"], info["scale"]),
        )
        self.streams = StreamContainer([self._stream])
        self.duration = info["frames"] * info["scale"] * AV_TIME_BASE // info["rate"]
        self.start_time = 0
        self.bit_rate = (
            0 if not self.duration else self.size * 8 * AV_TIME_BASE // self.duration
        )
        raw = np.frombuffer(self._data, dtype=np.uint8)
        capacity = max(1, (len(self._data) - 32) // 12 + 1)
        offsets = np.empty(capacity, dtype=np.int64)
        sizes = np.empty(capacity, dtype=np.int64)
        timestamps = np.empty(capacity, dtype=np.int64)
        keyframes = np.empty(capacity, dtype=np.int64)
        count = lib().mav_scan_ivf(
            addr(raw),
            raw.size,
            addr(offsets),
            addr(sizes),
            addr(timestamps),
            addr(keyframes),
            capacity,
        )
        if count < 0:
            raise ValueError("truncated or malformed IVF packet table")
        self._packet_offsets = memoryview(offsets)[:count]
        self._packet_sizes = memoryview(sizes)[:count]
        self._packet_pts = memoryview(timestamps)[:count]
        self._packet_durations = None
        self._keyframes = memoryview(keyframes)[:count]

    def _selected(self, args, kwargs) -> bool:
        if not args and not kwargs:
            return True
        for value in args:
            if value is self._stream or value == self._stream.index:
                return True
        for stream_type, index in kwargs.items():
            if stream_type == self._stream.type and index in (None, 0, self._stream):
                return True
        return False

    def demux(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("container is closed")
        if not self._selected(args, kwargs):
            return
        count = len(self._packet_offsets)
        ivf = self.format.name == "ivf"
        time_base = self._stream.time_base
        data = self._data
        stream = self._stream
        offsets = self._packet_offsets
        sizes = self._packet_sizes
        timestamps = self._packet_pts
        durations = self._packet_durations
        keyframes = self._keyframes
        packet_new = Packet.__new__
        for index in range(count):
            offset = offsets[index]
            packet = packet_new(Packet)
            packet._data = data
            packet._offset = offset + 12 if ivf else offset
            packet._size = sizes[index]
            packet.pts = timestamps[index]
            packet.dts = packet.pts
            packet.duration = 1 if durations is None else durations[index]
            packet._time_base = time_base
            packet._stream = stream
            packet._pos = offset
            packet.is_keyframe = True if keyframes is None else bool(keyframes[index])
            packet.is_corrupt = False
            packet.opaque = None
            yield packet
        flush = Packet()
        flush.time_base = time_base
        flush.stream = stream
        yield flush

    def decode(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError("codec decoding is outside mojo-av's covered subset")

    def close(self) -> None:
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False


def _detect_format(data: bytes, requested: str | None) -> str:
    if requested:
        name = requested.lower()
        if name in {"wav", "wave"}:
            return "wav"
        if name == "ivf":
            return "ivf"
        return name
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:4] == b"DKIF":
        return "ivf"
    raise ValueError("could not determine input format")


def _parse_wav(data: bytes) -> dict[str, int | str]:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("invalid RIFF/WAVE header")
    fmt = None
    data_chunk = None
    riff_end = struct.unpack_from("<I", data, 4)[0] + 8
    if riff_end < 12 or riff_end > len(data):
        raise ValueError("truncated RIFF payload")
    position = 12
    while position + 8 <= riff_end:
        chunk_id = data[position : position + 4]
        chunk_size = struct.unpack_from("<I", data, position + 4)[0]
        start = position + 8
        end = start + chunk_size
        if end > riff_end:
            raise ValueError("truncated RIFF chunk")
        if chunk_id == b"fmt ":
            fmt = data[start:end]
        elif chunk_id == b"data" and data_chunk is None:
            data_chunk = (start, chunk_size)
        position = end + (chunk_size & 1)
    if fmt is None or len(fmt) < 16 or data_chunk is None:
        raise ValueError("WAV requires fmt and data chunks")
    tag, channels, sample_rate, _, block_align, bits = struct.unpack_from(
        "<HHIIHH", fmt
    )
    if tag == 0xFFFE and len(fmt) >= 40:
        tag = struct.unpack_from("<H", fmt, 24)[0]
    if channels <= 0 or sample_rate <= 0 or block_align <= 0:
        raise ValueError("invalid WAV audio format")
    bytes_per_sample = (bits + 7) // 8
    if bits <= 0 or block_align != channels * bytes_per_sample:
        raise ValueError("inconsistent WAV sample layout")
    if data_chunk[1] % block_align:
        raise ValueError("WAV data ends with a partial sample frame")
    codecs = {
        (1, 8): ("pcm_u8", "u8"),
        (1, 16): ("pcm_s16le", "s16"),
        (1, 24): ("pcm_s24le", "s32"),
        (1, 32): ("pcm_s32le", "s32"),
        (3, 32): ("pcm_f32le", "flt"),
        (3, 64): ("pcm_f64le", "dbl"),
    }
    try:
        codec, sample_format = codecs[(tag, bits)]
    except KeyError:
        raise ValueError(f"unsupported WAV format tag={tag}, bits={bits}") from None
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "block_align": block_align,
        "codec": codec,
        "sample_format": sample_format,
        "data_start": data_chunk[0],
        "data_size": data_chunk[1],
    }


def _parse_ivf_header(data: bytes) -> dict[str, int]:
    if len(data) < 32 or data[:4] != b"DKIF":
        raise ValueError("invalid IVF header")
    version, header_size = struct.unpack_from("<HH", data, 4)
    fourcc = data[8:12]
    width, height, rate, scale, frames = struct.unpack_from("<HHIII", data, 12)
    if version != 0 or header_size != 32:
        raise ValueError("unsupported IVF header version")
    if fourcc != b"VP80":
        raise ValueError("only IVF VP8 streams are supported")
    if width <= 0 or height <= 0 or rate <= 0 or scale <= 0:
        raise ValueError("invalid IVF stream parameters")
    return {
        "width": width,
        "height": height,
        "rate": rate,
        "scale": scale,
        "frames": frames,
    }


builtins_open = open


def open(file, mode: str = "r", **kwargs) -> InputContainer:
    """Open a WAV or IVF input, matching ``av.open(file, mode='r', **kwargs)``."""
    if mode not in {"r", "rb"}:
        raise ValueError("mojo-av's covered subset supports input mode only")
    return InputContainer(file, **kwargs)
