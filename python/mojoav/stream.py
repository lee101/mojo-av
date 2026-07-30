from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass
class CodecContext:
    name: str
    width: int = 0
    height: int = 0
    sample_rate: int = 0
    channels: int = 0
    format: str | None = None


class Stream:
    def __init__(
        self,
        *,
        type: str,
        time_base: Fraction,
        codec_context: CodecContext,
        duration: int,
        frames: int = 0,
        start_time: int | None = None,
        average_rate: Fraction | None = None,
        base_rate: Fraction | None = None,
    ):
        self.index = 0
        self.type = type
        self.time_base = time_base
        self.codec_context = codec_context
        self.duration = duration
        self.frames = frames
        self.start_time = start_time
        self.average_rate = average_rate
        self.base_rate = base_rate
        self.metadata: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self.codec_context.name

    def __repr__(self) -> str:
        return f"<mojoav.{self.type.title()}Stream #{self.index} {self.name}>"


class StreamContainer:
    def __init__(self, streams):
        self._streams = tuple(streams)

    @property
    def audio(self):
        return tuple(stream for stream in self if stream.type == "audio")

    @property
    def video(self):
        return tuple(stream for stream in self if stream.type == "video")

    def get(self, *types, **indices):
        selected = list(self._streams)
        if types:
            selected = [stream for stream in selected if stream.type in types]
        for stream_type, index in indices.items():
            matches = [stream for stream in selected if stream.type == stream_type]
            if index is not None:
                matches = [matches[index]] if -len(matches) <= index < len(matches) else []
            selected = matches
        return selected

    def __iter__(self):
        return iter(self._streams)

    def __len__(self) -> int:
        return len(self._streams)

    def __getitem__(self, index):
        return self._streams[index]
