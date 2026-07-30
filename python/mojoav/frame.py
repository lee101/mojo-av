from __future__ import annotations

from fractions import Fraction


class Frame:
    """Timing-focused base class compatible with PyAV's frame attributes."""

    def __init__(self):
        self.pts: int | None = None
        self.dts: int | None = None
        self.duration: int = 0
        self._time_base: Fraction | None = None
        self.is_corrupt = False
        self.key_frame = False
        self.opaque = None

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

    @property
    def time(self) -> float | None:
        if self.pts is None:
            return None
        if self._time_base is None:
            return float("nan")
        return float(self.pts * self._time_base)

    def __repr__(self) -> str:
        return f"<mojoav.{type(self).__name__}, pts={self.pts} at 0x{id(self):x}>"


class VideoFrame(Frame):
    def __init__(self, width: int = 0, height: int = 0, format: str = "yuv420p"):
        super().__init__()
        self.width = int(width)
        self.height = int(height)
        self.format = format


class AudioFrame(Frame):
    def __init__(
        self, format: str = "s16", layout: str = "stereo", samples: int = 0
    ):
        super().__init__()
        self.format = format
        self.layout = layout
        self.samples = int(samples)
        self.sample_rate = 0

    @property
    def rate(self) -> int:
        return self.sample_rate

    @rate.setter
    def rate(self, value: int) -> None:
        self.sample_rate = int(value)
