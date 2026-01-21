from __future__ import annotations

from typing import Generator, Iterable, Optional

from app.domain.entities.audio import AudioFrame
from app.domain.services.tts_port import ITTSStreamer
from app.domain.value_objects.types import LanguageCode
from app.infrastructure.tts.piper_streamer import PiperTTSAdapter
from app.infrastructure.tts.silero_streamer import SileroTTSAdapter


class MultiTTSAdapter(ITTSStreamer):
    """
    Delegates to Silero when available/supported, otherwise falls back to Piper.
    Currently routes EN_US to Silero if configured; all others use Piper.
    """
    def __init__(self, piper: PiperTTSAdapter, silero: Optional[SileroTTSAdapter] = None):
        self.piper = piper
        self.silero = silero

    def stream_pcm(
        self,
        text_or_iter: str | Iterable[str],
        language: LanguageCode,
        quality: str,
        lesson:str
    ) -> Generator[AudioFrame, None, None]:
        if self.silero and language == LanguageCode.EN_US and quality == 'high':
            # Silero path; let exceptions propagate to fail fast (no silent fallback)
            yield from self.silero.stream_pcm(text_or_iter, language)
            return
        yield from self.piper.stream_pcm(text_or_iter, language)
