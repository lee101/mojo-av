"""A Mojo-accelerated subset of PyAV container demux and timing APIs."""

from .container import InputContainer, open
from .frame import AudioFrame, Frame, VideoFrame
from .packet import Packet
from .timing import NOPTS_VALUE, durations_from_pts, rescale, rescale_many

time_base = 1_000_000
__version__ = "0.1.0"

__all__ = [
    "AudioFrame",
    "Frame",
    "InputContainer",
    "NOPTS_VALUE",
    "Packet",
    "VideoFrame",
    "durations_from_pts",
    "open",
    "rescale",
    "rescale_many",
    "time_base",
]
