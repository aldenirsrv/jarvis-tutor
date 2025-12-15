from __future__ import annotations

from dataclasses import dataclass
from app.domain.value_objects.types import AudioFormat


@dataclass
class AudioFrame:
    sample_rate: int
    channels: int
    sample_width: int
    data: bytes


@dataclass
class AudioStream:
    format: AudioFormat

