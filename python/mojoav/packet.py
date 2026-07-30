from __future__ import annotations

from fractions import Fraction


class Packet:
    """A packet of encoded bytes with PyAV-compatible timing attributes."""

    __slots__ = (
        "_data",
        "_offset",
        "_size",
        "pts",
        "dts",
        "duration",
        "_time_base",
        "_stream",
        "_pos",
        "is_keyframe",
        "is_corrupt",
        "opaque",
    )

    def __init__(self, input=None):
        if input is None:
            self._data = b""
        elif isinstance(input, int):
            self._data = bytes(input)
        elif isinstance(input, str):
            self._data = input.encode()
        else:
            self._data = bytes(input)
        self._offset = 0
        self._size = len(self._data)
        self.pts: int | None = None
        self.dts: int | None = None
        self.duration: int = 0
        self._time_base: Fraction | None = None
        self._stream = None
        self._pos: int | None = None
        self.is_keyframe = False
        self.is_corrupt = False
        self.opaque = None

    @classmethod
    def _from_demux(
        cls,
        data: bytes,
        offset: int,
        size: int,
        pts: int,
        duration: int,
        time_base: Fraction,
        stream,
        pos: int,
        is_keyframe: bool,
    ):
        packet = cls.__new__(cls)
        packet._data = data
        packet._offset = offset
        packet._size = size
        packet.pts = pts
        packet.dts = pts
        packet.duration = duration
        packet._time_base = time_base
        packet._stream = stream
        packet._pos = pos
        packet.is_keyframe = is_keyframe
        packet.is_corrupt = False
        packet.opaque = None
        return packet

    @property
    def size(self) -> int:
        return self._size

    @property
    def buffer_size(self) -> int:
        return self._size

    @property
    def pos(self) -> int | None:
        return self._pos

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, value) -> None:
        self._stream = value

    @property
    def stream_index(self) -> int:
        return 0 if self._stream is None else self._stream.index

    @property
    def time_base(self) -> Fraction | None:
        return self._time_base

    @time_base.setter
    def time_base(self, value) -> None:
        numerator = value.numerator
        denominator = value.denominator
        self._time_base = (
            None if numerator == 0 else Fraction(int(numerator), int(denominator))
        )

    def decode(self):
        raise NotImplementedError("codec decoding is outside mojo-av's covered subset")

    def __bytes__(self) -> bytes:
        if self._offset == 0 and self._size == len(self._data):
            return self._data
        return self._data[self._offset : self._offset + self._size]

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        index = self.stream_index
        return (
            f"<mojoav.Packet of #{index}, dts={self.dts}, pts={self.pts}; "
            f"{self.size} bytes at 0x{id(self):x}>"
        )
