from fractions import Fraction
import math

import av
import numpy as np
import pytest

import mojoav


def reference_rescale(value, src, dst, rounding):
    ratio = Fraction(value) * Fraction(src) / Fraction(dst)
    numerator, denominator = ratio.numerator, ratio.denominator
    if rounding == "zero":
        return (
            numerator // denominator
            if numerator >= 0
            else -((-numerator) // denominator)
        )
    if rounding == "near":
        magnitude = (abs(numerator) + denominator // 2) // denominator
        return magnitude if numerator >= 0 else -magnitude
    if rounding == "down":
        return numerator // denominator
    return -((-numerator) // denominator)


@pytest.mark.parametrize("rounding", ["zero", "near", "down", "up"])
@pytest.mark.parametrize(
    ("src", "dst"),
    [
        (Fraction(1, 30), Fraction(1, 1000)),
        (Fraction(1001, 30000), Fraction(1, 90000)),
        (Fraction(1, 48000), Fraction(1, 1_000_000)),
    ],
)
def test_rescale_many_matches_exact_reference(rounding, src, dst):
    values = np.arange(-1003, 1004, dtype=np.int64)
    actual = mojoav.rescale_many(values, src, dst, rounding)
    expected = np.array(
        [reference_rescale(int(value), src, dst, rounding) for value in values],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_rescale_random_large_values():
    rng = np.random.default_rng(12)
    values = rng.integers(-(1 << 42), 1 << 42, size=10_000, dtype=np.int64)
    src = Fraction(1001, 60000)
    dst = Fraction(1, 90_000)
    actual = mojoav.rescale_many(values, src, dst)
    expected = np.array(
        [reference_rescale(int(value), src, dst, "near") for value in values],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(actual, expected)


def test_rescale_scalar_and_none():
    assert mojoav.rescale(3, Fraction(1, 2), Fraction(1, 1)) == 2
    assert mojoav.rescale(-3, Fraction(1, 2), Fraction(1, 1)) == -2
    assert mojoav.rescale(None, Fraction(1, 2), Fraction(1, 1)) is None


def test_rescale_preserves_nopts():
    values = np.array([mojoav.NOPTS_VALUE, 10], dtype=np.int64)
    actual = mojoav.rescale_many(values, Fraction(1, 10), Fraction(1, 100))
    assert actual.tolist() == [mojoav.NOPTS_VALUE, 100]


@pytest.mark.parametrize(
    ("values", "final", "expected"),
    [
        ([0, 3, 9], 4, [3, 6, 4]),
        ([7], 2, [2]),
        ([], 0, []),
        ([0, mojoav.NOPTS_VALUE, 5], 1, [mojoav.NOPTS_VALUE, mojoav.NOPTS_VALUE, 1]),
    ],
)
def test_durations_from_pts(values, final, expected):
    actual = mojoav.durations_from_pts(values, final)
    assert actual.tolist() == expected


@pytest.mark.parametrize("count", [2, 7, 8, 9, 17, 19])
def test_durations_simd_tail(count):
    values = np.arange(count, dtype=np.int64)
    values[count // 2] = mojoav.NOPTS_VALUE
    expected = np.ones(count, dtype=np.int64)
    expected[count // 2 - 1 : count // 2 + 1] = mojoav.NOPTS_VALUE
    expected[-1] = 7
    np.testing.assert_array_equal(mojoav.durations_from_pts(values, 7), expected)


def test_durations_parallel_threshold():
    count = 1_000_003
    values = np.arange(count, dtype=np.int64)
    missing = 125_001
    values[missing] = mojoav.NOPTS_VALUE
    expected = np.ones(count, dtype=np.int64)
    expected[missing - 1 : missing + 1] = mojoav.NOPTS_VALUE
    expected[-1] = 11
    np.testing.assert_array_equal(mojoav.durations_from_pts(values, 11), expected)


@pytest.mark.parametrize("cls_name", ["VideoFrame", "AudioFrame"])
def test_frame_timing_matches_pyav(cls_name):
    reference = getattr(av, cls_name)()
    actual = getattr(mojoav, cls_name)()
    for obj in (reference, actual):
        obj.time_base = Fraction(1001, 30000)
        obj.pts = 123
        obj.dts = 120
        obj.duration = 1
    assert actual.pts == reference.pts
    assert actual.dts == reference.dts
    assert actual.duration == reference.duration
    assert actual.time_base == reference.time_base
    assert actual.time == reference.time


@pytest.mark.parametrize("cls_name", ["VideoFrame", "AudioFrame"])
def test_frame_missing_timing_matches_pyav(cls_name):
    reference = getattr(av, cls_name)()
    actual = getattr(mojoav, cls_name)()
    assert actual.pts == reference.pts
    assert actual.dts == reference.dts
    assert actual.duration == reference.duration
    assert actual.time_base == reference.time_base
    assert actual.time == reference.time
    for obj in (reference, actual):
        obj.pts = 4
    assert math.isnan(actual.time)
    assert math.isnan(reference.time)


def test_packet_constructor_and_timing_match_pyav():
    reference = av.Packet(b"encoded")
    actual = mojoav.Packet(b"encoded")
    for packet in (reference, actual):
        packet.pts = 90
        packet.dts = 80
        packet.duration = 10
        packet.time_base = Fraction(1, 90000)
        packet.is_keyframe = True
    attributes = (
        "pts",
        "dts",
        "duration",
        "time_base",
        "size",
        "buffer_size",
        "is_keyframe",
        "stream_index",
    )
    assert [getattr(actual, name) for name in attributes] == [
        getattr(reference, name) for name in attributes
    ]
    assert bytes(actual) == bytes(reference)


def test_timing_validation():
    with pytest.raises(ValueError):
        mojoav.rescale_many([1], Fraction(0), Fraction(1))
    with pytest.raises(ValueError):
        mojoav.rescale_many([1], Fraction(1), Fraction(1), "bankers")
    with pytest.raises(ValueError):
        mojoav.rescale_many(np.zeros((2, 2), dtype=np.int64), 1, 1)
    with pytest.raises(TypeError):
        mojoav.rescale_many([1.25], 1, 1)
    with pytest.raises(TypeError):
        mojoav.durations_from_pts([1, 2], 1.5)


def test_integer_inputs_are_exact_and_bounded():
    assert mojoav.rescale_many((value for value in [1, 2]), 1, 1).tolist() == [1, 2]
    assert mojoav.rescale_many(np.array([1], dtype=np.uint64), 1, 1).tolist() == [1]
    with pytest.raises(OverflowError):
        mojoav.rescale_many(np.array([1 << 63], dtype=np.uint64), 1, 1)
    with pytest.raises(OverflowError):
        mojoav.durations_from_pts([1 << 63])


def test_rescale_rejects_factor_overflow():
    limit = np.iinfo(np.int64).max
    with pytest.raises(OverflowError):
        mojoav.rescale_many([1], Fraction(limit, 1), Fraction(1, limit))


def test_rescale_rejects_result_overflow():
    limit = np.iinfo(np.int64).max
    with pytest.raises(OverflowError):
        mojoav.rescale_many([limit], Fraction(2, 1), Fraction(1, 1))
