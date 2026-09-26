# mojo-av

`mojo-av` is a standalone Mojo port of a focused, useful subset of
[PyAV](https://github.com/PyAV-Org/PyAV): container demux and frame/packet
timing math. Import it as `mojoav` and, for the covered API, use the same names
and call shapes as PyAV.

This is not a wrapper around PyAV. The demuxers, packet objects, and timing
kernels are implemented in this repository. PyAV is a development dependency
used as the behavioral reference in parity tests and benchmarks.

## Covered subset

- `open(file, mode="r", format=None)` and context-managed `InputContainer`
- RIFF/WAVE demux for 8-, 16-, 24-, and 32-bit integer PCM and 32-/64-bit
  floating-point PCM
- IVF demux for VP8 streams
- `InputContainer.demux(*streams, **stream_selection)`
- `Packet` payload, PTS, DTS, duration, time base, position, stream, and
  keyframe metadata
- `Frame`, `VideoFrame`, and `AudioFrame` timing attributes, including
  `frame.time`
- Exact integer `rescale` and batched `rescale_many` with FFmpeg-style
  nearest, zero, floor, and ceiling rounding
- Batched duration derivation from presentation timestamps

The implementation intentionally does not cover codecs or frame decoding,
encoding, muxing, seeking, arbitrary FFmpeg container formats, devices,
filters, hardware acceleration, image planes, or audio sample buffers.
Unsupported decoding calls raise `NotImplementedError`; unsupported formats
raise `ValueError`, and unsupported `open` options raise `TypeError`.

## Install

Install the pinned Mojo nightly and all Python dependencies, then build the
shared library:

```bash
pixi install
pixi run build
```

Run the parity suite with:

```bash
pixi run test
```

## Usage

The normal PyAV-style demux loop works for the supported containers:

```python
import mojoav as av

with av.open("audio.wav") as container:
    stream = container.streams.audio[0]
    print(stream.time_base)
    for packet in container.demux(stream):
        if packet.size:
            print(packet.pts, packet.duration, packet.time_base, packet.size)
```

Timestamp arrays stay integer-exact:

```python
from fractions import Fraction
import mojoav as av

milliseconds = av.rescale_many(
    [0, 1, 2, 3],
    src_time_base=Fraction(1, 30),
    dst_time_base=Fraction(1, 1000),
)
assert milliseconds.tolist() == [0, 33, 67, 100]
```

With `PYTHONPATH=python` configured by Pixi, save either example and run it
with `pixi run python example.py`.

## Benchmarks

Measured with `pixi run bench` on a dual-socket Intel Xeon E5-2697 v4 system
(36 physical cores, 72 threads), Linux 6.8.0, Python 3.13.14, and PyAV 18.0.0.
Times are the best of three or five runs. A speedup above 1 means mojo-av was
faster.

| Workload | mojo-av | Reference | Speedup | Compared with |
|---|---:|---:|---:|---|
| rescale 1M timestamps | 37.24 ms | 487.52 ms | 13.09x | exact Python integer loop |
| derive 5M durations | 16.81 ms | 42.53 ms | 2.53x | NumPy `diff` |
| demux 200k IVF packets | 115.55 ms | 482.87 ms | 4.18x | PyAV / FFmpeg |

The benchmark validates every timing result and the total demuxed byte count
before printing. These are measurements from this machine, not projections;
different CPUs and media packet sizes will produce different results.

No GPU path is provided. Duration derivation performs about one subtraction per
24 bytes moved, IVF scanning is serial and metadata-write-heavy, and timestamp
rescaling is also below the roughly two-operations-per-byte cutoff. Transfer and
launch overhead would dominate these kernels.

## How it works

`src/capi.mojo` is one compilation unit that exports an allocation-free C ABI.
`build/build.sh` compiles it to `dist/libmojo-av.so`. The Python layer loads
that library with `ctypes`.

Python owns file bytes, NumPy metadata arrays, and all returned objects.
Contiguous array addresses cross the ABI as 64-bit integers and Mojo rebuilds
typed pointers with `AnyOrigin[mut=True]`. Mojo scans IVF record headers,
generates fixed-size WAV packet tables, rescales rational timestamps, and
derives durations. Duration derivation uses native-width SIMD with a scalar
remainder. At one million differences or more it splits the difference range
into sixteen parts, each run on the calling thread: the kernel moves 16 bytes
per difference and is bandwidth-bound, so worker threads measured slower.
Demuxed packets retain the container bytes without copying and materialize a
payload slice only when its bytes are requested, so there is no cross-language
allocator or lifetime protocol.

WAV parsing honors RIFF chunk padding and does not assume that `fmt ` precedes
`data`. IVF parsing validates every record boundary before exposing a packet.
Timestamp rescaling reduces rational factors before applying signed,
integer-only rounding and preserves the standard `INT64_MIN` no-timestamp
sentinel.

## Development

All supported workflows run through Pixi:

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task takes a machine-wide file lock to avoid overlapping factory
benchmark jobs.

## License

MIT
