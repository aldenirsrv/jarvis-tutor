from __future__ import annotations

from typing import Generator, Iterable, Callable
import logging
from app.application.dtos.tts_input import TTSRequestDTO
from app.domain.entities.audio import AudioFrame
from app.domain.services.chat_port import IChatStreamer
from app.domain.services.tts_port import ITTSStreamer


class StreamTTSUseCase:
    def __init__(
        self,
        chat: IChatStreamer,
        tts: ITTSStreamer,
        chunker: Callable[[Iterable[str]], Iterable[str]],
    ):
        self.chat = chat
        self.tts = tts
        self.chunker = chunker

    def execute(self, dto: TTSRequestDTO) -> Generator[AudioFrame, None, None]:
        logger = logging.getLogger(__name__)

        logger.info("chat stream start | lang=%s", dto.language.value)
        text_iter = self.chat.stream(dto.message, dto.language)
        logger.info("chat stream got iterator | lang=%s", dto.language.value)

        # Choose chunker language based on ISO code; default to English if unsupported by pysbd
        lang_prefix = dto.language_iso.value if dto.language_iso.value in ("en", "pt", "fr", "es", "de", "it") else "en"
        buffered_iter = self.chunker(text_iter, language=lang_prefix)

        token_count = 0
        char_count = 0

        def counted_iter():
            nonlocal token_count, char_count
            for piece in buffered_iter:
                t = piece or ""
                token_count += len(t.split())
                char_count += len(t)
                yield piece

        for frame in self.tts.stream_pcm(counted_iter(), dto.language):
            logger.info("tts frame yielded | sr=%d ch=%d sw=%d bytes=%d", frame.sample_rate, frame.channels, frame.sample_width, len(frame.data))
            yield frame
        logger.info(
            "tts stream summary | lang=%s | tokens=%d | chars=%d",
            dto.language.value,
            token_count,
            char_count,
        )
