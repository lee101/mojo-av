import io
import struct
import wave
from fractions import Fraction

import av
import pytest

import mojoav


def make_wav(channels=2, sample_width=2, rate=48000, frames=6400):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(sample_width)
        output.setframerate(rate)
        output.writeframes(bytes((index * 17) & 255 for index in range(frames * channels * sample_width)))
    return buffer.getvalue()


def make_ivf():
    payloads = [
        (0, b"\x10\x00\x00\x9d\x01\x2a\x10\x00\x10\x00abc"),
        (2, b"\x11\x00\x00inter"),
        (5, b"\x10\x00\x00\x9d\x01\x2a\x10\x00\x10\x00xyz"),
    ]
    header = struct.pack(
        "<4sHH4sHHIIII",
        b"DKIF",
        0,
        32,
        b"VP80",
        16,
        16,
        30,
        1,
        len(payloads),
        0,
    )
    records = b"".join(
        struct.pack("<IQ", len(payload), timestamp) + payload
        for timestamp, payload in payloads
    )
    return header + records


def make_raw_wav(tag, bits, channels=2, rate=48000, frames=5000):
    block_align = channels * bits // 8
    payload = bytes(
        (index * 17) & 255 for index in range(frames * block_align)
    )
    fmt = struct.pack(
        "<HHIIHH",
        tag,
        channels,
        rate,
        rate * block_align,
        block_align,
        bits,
    )
    body = (
        b"JUNK"
        + struct.pack("<I", 1)
        + b"x\x00"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(payload))
        + payload
    )
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def packet_record(packet):
    return {
        "data": bytes(packet),
        "pts": packet.pts,
        "dts": packet.dts,
        "duration": packet.duration,
        "time_base": packet.time_base,
        "pos": packet.pos,
        "size": packet.size,
        "key": packet.is_keyframe,
        "stream_index": packet.stream_index,
    }


@pytest.mark.parametrize(
    ("channels", "sample_width", "rate", "frames"),
    [(1, 1, 8000, 1), (1, 2, 44100, 4097), (2, 2, 48000, 6400)],
)
def test_wav_demux_packet_parity(channels, sample_width, rate, frames):
    data = make_wav(channels, sample_width, rate, frames)
    reference = av.open(io.BytesIO(data))
    actual = mojoav.open(io.BytesIO(data))
    assert [packet_record(packet) for packet in actual.demux()] == [
        packet_record(packet) for packet in reference.demux()
    ]


@pytest.mark.parametrize(
    ("channels", "sample_width", "rate", "codec"),
    [(1, 1, 8000, "pcm_u8"), (1, 2, 44100, "pcm_s16le"), (2, 2, 48000, "pcm_s16le")],
)
def test_wav_container_and_stream_parity(channels, sample_width, rate, codec):
    data = make_wav(channels, sample_width, rate, 5000)
    reference = av.open(io.BytesIO(data))
    actual = mojoav.open(io.BytesIO(data))
    assert actual.format.name == reference.format.name
    assert actual.duration == reference.duration
    assert actual.start_time == reference.start_time
    assert actual.size == reference.size
    ref_stream = reference.streams.audio[0]
    stream = actual.streams.audio[0]
    assert len(actual.streams) == len(reference.streams) == 1
    assert stream.index == ref_stream.index
    assert stream.type == ref_stream.type
    assert stream.time_base == ref_stream.time_base == Fraction(1, rate)
    assert stream.duration == ref_stream.duration
    assert stream.start_time == ref_stream.start_time
    assert stream.frames == ref_stream.frames
    assert stream.codec_context.name == ref_stream.codec_context.name == codec
    assert stream.codec_context.sample_rate == ref_stream.codec_context.sample_rate
    assert stream.codec_context.channels == ref_stream.codec_context.channels


@pytest.mark.parametrize(
    ("tag", "bits", "codec", "sample_format"),
    [
        (1, 24, "pcm_s24le", "s32"),
        (1, 32, "pcm_s32le", "s32"),
        (3, 32, "pcm_f32le", "flt"),
        (3, 64, "pcm_f64le", "dbl"),
    ],
)
def test_additional_wav_formats_match_pyav(tag, bits, codec, sample_format):
    data = make_raw_wav(tag, bits)
    reference = av.open(io.BytesIO(data))
    actual = mojoav.open(io.BytesIO(data))
    assert [packet_record(packet) for packet in actual.demux()] == [
        packet_record(packet) for packet in reference.demux()
    ]
    stream = actual.streams.audio[0]
    assert stream.codec_context.name == reference.streams.audio[0].codec_context.name == codec
    assert stream.codec_context.format == reference.streams.audio[0].codec_context.format.name == sample_format


def test_ivf_demux_packet_parity():
    data = make_ivf()
    reference = av.open(io.BytesIO(data))
    actual = mojoav.open(io.BytesIO(data))
    assert [packet_record(packet) for packet in actual.demux()] == [
        packet_record(packet) for packet in reference.demux()
    ]


def test_demux_packet_payload_is_lazy_and_zero_copy():
    data = make_ivf()
    container = mojoav.open(io.BytesIO(data))
    packets = list(container.demux())
    packet = packets[0]
    assert len({id(item) for item in packets}) == len(packets)
    assert packet._data is container._data
    assert bytes(packet) == data[44 : 44 + packet.size]


def test_ivf_container_and_stream_parity():
    data = make_ivf()
    reference = av.open(io.BytesIO(data))
    actual = mojoav.open(io.BytesIO(data))
    assert actual.format.name == reference.format.name
    assert actual.duration == reference.duration
    assert actual.start_time == reference.start_time
    assert actual.bit_rate == reference.bit_rate
    assert actual.size == reference.size
    stream = actual.streams.video[0]
    ref_stream = reference.streams.video[0]
    for name in ("index", "type", "time_base", "duration", "frames", "start_time", "base_rate"):
        assert getattr(stream, name) == getattr(ref_stream, name)
    assert stream.average_rate == ref_stream.average_rate
    assert stream.codec_context.name == ref_stream.codec_context.name
    assert stream.codec_context.width == ref_stream.codec_context.width
    assert stream.codec_context.height == ref_stream.codec_context.height


@pytest.mark.parametrize("selection", ["stream", "index", "keyword"])
def test_demux_stream_selection(selection):
    container = mojoav.open(io.BytesIO(make_ivf()))
    if selection == "stream":
        packets = list(container.demux(container.streams.video[0]))
    elif selection == "index":
        packets = list(container.demux(0))
    else:
        packets = list(container.demux(video=0))
    assert len(packets) == 4
    assert list(container.demux(audio=0)) == []


def test_open_path_and_context_manager(tmp_path):
    path = tmp_path / "audio.wav"
    path.write_bytes(make_wav())
    with mojoav.open(path) as container:
        assert container.name == str(path)
        assert len(list(container.demux())) == 3
    with pytest.raises(RuntimeError, match="closed"):
        list(container.demux())


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"not a media file",
        b"DKIF" + bytes(28),
        make_ivf()[:-1],
        b"RIFF" + bytes(4) + b"WAVE",
    ],
)
def test_malformed_inputs_raise(data):
    with pytest.raises(ValueError):
        mojoav.open(io.BytesIO(data))


def test_format_override_and_write_rejection():
    data = make_wav()
    assert mojoav.open(io.BytesIO(data), format="wav").format.name == "wav"
    with pytest.raises(ValueError, match="input mode"):
        mojoav.open(io.BytesIO(data), mode="w")
    with pytest.raises(TypeError, match="unsupported open option"):
        mojoav.open(io.BytesIO(data), buffer_size=1024)


def test_wav_rejects_inconsistent_or_truncated_layout():
    data = bytearray(make_wav(channels=2, sample_width=2, frames=4))
    fmt_start = data.index(b"fmt ") + 8
    struct.pack_into("<H", data, fmt_start + 12, 1)
    with pytest.raises(ValueError, match="sample layout"):
        mojoav.open(io.BytesIO(data))

    data = bytearray(make_wav(channels=1, sample_width=2, frames=4)[:-1])
    struct.pack_into("<I", data, 4, len(data) - 8)
    data_size_offset = data.index(b"data") + 4
    struct.pack_into("<I", data, data_size_offset, 7)
    with pytest.raises(ValueError, match="partial sample frame"):
        mojoav.open(io.BytesIO(data))


def test_decode_is_explicitly_unsupported():
    container = mojoav.open(io.BytesIO(make_ivf()))
    packet = next(container.demux())
    with pytest.raises(NotImplementedError, match="decoding"):
        packet.decode()
    with pytest.raises(NotImplementedError, match="decoding"):
        container.decode(video=0)
