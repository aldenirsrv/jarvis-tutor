from __future__ import annotations

from typing import Generator, Iterable
from app.domain.entities.audio import AudioFrame
from app.domain.services.tts_port import ITTSStreamer
from app.domain.value_objects.types import LanguageCode
from app.infrastructure.config.voices import VoiceRegistry
from app.infrastructure.tts.piper_streaming import PiperStreamer
import logging

logger = logging.getLogger(__name__)

class PiperTTSAdapter(ITTSStreamer):
    def __init__(self, registry: VoiceRegistry):
        self.registry = registry

    def stream_pcm(self, text_or_iter: str | Iterable[str], language: LanguageCode) -> Generator[AudioFrame, None, None]:
        voice, cfg = self.registry.get(language.value)
        streamer = PiperStreamer(voice, cfg, first_chunk_cb=None)

        for sr, ch, sw, pcm in streamer.stream_pcm(text_or_iter):
            logger.info("piper frame yielded | sr=%d ch=%d sw=%d bytes=%d", sr, ch, sw, len(pcm))
            yield AudioFrame(sample_rate=sr, channels=ch, sample_width=sw, data=pcm)
