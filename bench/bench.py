"""mojo-av benchmarks against PyAV and direct Python/NumPy equivalents."""

from __future__ import annotations

import io
import os
import platform
import struct
import sys
import time
from fractions import Fraction

import av
import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

import mojoav  # noqa: E402


def best_time(function, repetitions=3):
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        best = min(best, time.perf_counter() - start)
    return best, result


def reference_rescale(values, src, dst):
    ratio = src / dst
    numerator, denominator = ratio.numerator, ratio.denominator
    return np.fromiter(
        (
            ((abs(int(value)) * numerator + denominator // 2) // denominator)
            * (-1 if value < 0 else 1)
            for value in values
        ),
        dtype=np.int64,
        count=values.size,
    )


def make_ivf(packet_count):
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
        packet_count,
        0,
    )
    payload = b"\x11\x00\x00payload"
    record_header = struct.pack("<I", len(payload))
    return header + b"".join(
        record_header + struct.pack("<Q", index) + payload
        for index in range(packet_count)
    )


def demux_count(module, data):
    container = module.open(io.BytesIO(data))
    return sum(packet.size for packet in container.demux())


def row(name, mojo_seconds, reference_seconds, reference_name):
    speedup = reference_seconds / mojo_seconds
    print(
        f"| {name} | {mojo_seconds * 1e3:.2f} ms | "
        f"{reference_seconds * 1e3:.2f} ms | {speedup:.2f}x | {reference_name} |"
    )


def main():
    rng = np.random.default_rng(4)
    timestamps = rng.integers(-(1 << 38), 1 << 38, size=1_000_000, dtype=np.int64)
    src = Fraction(1001, 60000)
    dst = Fraction(1, 90000)
    mojo_time, mojo_scaled = best_time(
        lambda: mojoav.rescale_many(timestamps, src, dst), repetitions=5
    )
    python_time, python_scaled = best_time(
        lambda: reference_rescale(timestamps, src, dst), repetitions=3
    )
    np.testing.assert_array_equal(mojo_scaled, python_scaled)

    pts = np.cumsum(
        rng.integers(1, 5, size=5_000_000, dtype=np.int64), dtype=np.int64
    )
    mojo_duration_time, mojo_durations = best_time(
        lambda: mojoav.durations_from_pts(pts, 1), repetitions=5
    )
    numpy_duration_time, numpy_durations = best_time(
        lambda: np.concatenate((np.diff(pts), np.array([1], dtype=np.int64))),
        repetitions=5,
    )
    np.testing.assert_array_equal(mojo_durations, numpy_durations)

    ivf = make_ivf(200_000)
    mojo_demux_time, mojo_bytes = best_time(
        lambda: demux_count(mojoav, ivf), repetitions=3
    )
    pyav_demux_time, pyav_bytes = best_time(
        lambda: demux_count(av, ivf), repetitions=3
    )
    assert mojo_bytes == pyav_bytes

    print(f"Machine: {platform.processor() or platform.machine()} ({platform.platform()})")
    print(f"Python {platform.python_version()}, PyAV {av.__version__}")
    print()
    print("| Workload | mojo-av | Reference | Speedup | Compared with |")
    print("|---|---:|---:|---:|---|")
    row("rescale 1M timestamps", mojo_time, python_time, "exact Python integer loop")
    row("derive 5M durations", mojo_duration_time, numpy_duration_time, "NumPy diff")
    row("demux 200k IVF packets", mojo_demux_time, pyav_demux_time, "PyAV / FFmpeg")


if __name__ == "__main__":
    main()
