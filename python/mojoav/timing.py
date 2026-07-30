from __future__ import annotations

from fractions import Fraction
import operator
from typing import Iterable

import numpy as np

from ._lib import addr, lib

NOPTS_VALUE = np.iinfo(np.int64).min
_ROUNDING = {"zero": 0, "near": 1, "down": 2, "up": 3}
_I64_MIN = np.iinfo(np.int64).min
_I64_MAX = np.iinfo(np.int64).max


def _integer_array(values, name: str) -> np.ndarray:
    """Convert integer values to contiguous int64 without lossy casts."""
    if not isinstance(values, np.ndarray):
        try:
            values = list(values)
        except TypeError:
            values = np.asarray(values)
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype.kind == "u":
        if array.size and int(array.max()) > _I64_MAX:
            raise OverflowError(f"{name} values must fit in int64")
    elif array.dtype.kind not in "ib":
        converted = []
        for value in array:
            try:
                integer = operator.index(value)
            except TypeError:
                raise TypeError(f"{name} values must be integers") from None
            if integer < _I64_MIN or integer > _I64_MAX:
                raise OverflowError(f"{name} values must fit in int64")
            converted.append(integer)
        array = np.asarray(converted, dtype=np.int64)
    return np.ascontiguousarray(array, dtype=np.int64)


def _time_base(value) -> Fraction:
    result = Fraction(value)
    if result <= 0:
        raise ValueError("time base must be positive")
    limit = np.iinfo(np.int64).max
    if result.numerator > limit or result.denominator > limit:
        raise OverflowError("time base components must fit in int64")
    return result


def rescale_many(
    timestamps: Iterable[int] | np.ndarray,
    src_time_base,
    dst_time_base,
    rounding: str = "near",
) -> np.ndarray:
    """Rescale an integer timestamp array between rational time bases.

    ``NOPTS_VALUE`` is preserved. ``near`` matches FFmpeg's nearest,
    halfway-away-from-zero timestamp rounding.
    """
    try:
        rounding_code = _ROUNDING[rounding]
    except KeyError:
        raise ValueError("rounding must be 'zero', 'near', 'down', or 'up'") from None
    source = _integer_array(timestamps, "timestamps")
    src = _time_base(src_time_base)
    dst = _time_base(dst_time_base)
    result = np.empty_like(source)
    if not source.size:
        return result
    status = lib().mav_rescale_many(
        addr(source),
        addr(result),
        source.size,
        src.numerator,
        src.denominator,
        dst.numerator,
        dst.denominator,
        rounding_code,
    )
    if status:
        raise OverflowError("timestamp rescale factors are outside the supported int64 range")
    return result


def rescale(timestamp: int | None, src_time_base, dst_time_base, rounding: str = "near"):
    """Rescale one timestamp, preserving ``None``."""
    if timestamp is None:
        return None
    values = _integer_array([timestamp], "timestamp")
    return int(rescale_many(values, src_time_base, dst_time_base, rounding)[0])


def durations_from_pts(
    pts: Iterable[int] | np.ndarray, final_duration: int = 0
) -> np.ndarray:
    """Return adjacent PTS deltas and the caller-supplied final duration."""
    values = _integer_array(pts, "pts")
    result = np.empty_like(values)
    if values.size:
        try:
            final = operator.index(final_duration)
        except TypeError:
            raise TypeError("final_duration must be an integer") from None
        if final < _I64_MIN or final > _I64_MAX:
            raise OverflowError("final_duration must fit in int64")
        status = lib().mav_durations_from_pts(
            addr(values), addr(result), values.size, final
        )
        if status:
            raise RuntimeError("duration derivation failed")
    return result
