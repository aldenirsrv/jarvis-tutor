from __future__ import annotations

import struct
import shutil
import logging
import itertools
from typing import Iterable, Generator
from fastapi import APIRouter, Request, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
import asyncio, json
from app.application.dtos.tts_input import TTSRequestDTO
from app.application.use_cases.stream_tts import StreamTTSUseCase
from app.domain.entities.audio import AudioFrame
from app.domain.value_objects.types import AudioFormat
from app.infrastructure.audio.aac_encoder import aac_stream
from app.interface.api.schemas import TTSRequest
from app.infrastructure.config.languages import get_languages_json
from app.infrastructure.tts.piper_streaming import PiperStreamer
from app.application.policies.chunking import buffered
logger = logging.getLogger(__name__)

router = APIRouter()


# def _wav_header(sr: int, ch: int, sw: int) -> bytes:
#     byte_rate = sr * ch * sw
#     block_align = ch * sw
#     bps = sw * 8
#     data_size = 0xFFFFFFFF
#     riff_size = 0xFFFFFFFF
#     return (
#         b"RIFF"
#         + struct.pack("<I", riff_size)
#         + b"WAVEfmt "
#         + struct.pack("<I", 16)
#         + struct.pack("<H", 1)
#         + struct.pack("<H", ch)
#         + struct.pack("<I", sr)
#         + struct.pack("<I", byte_rate)
#         + struct.pack("<H", block_align)
#         + struct.pack("<H", bps)
#         + b"data"
#         + struct.pack("<I", data_size)
#     )

def _wav_header(sr: int, ch: int, sw: int) -> bytes:
    byte_rate = sr * ch * sw
    block_align = ch * sw
    bps = sw * 8

    # Streaming-friendly: declare sizes as 0 (unknown)
    # This avoids "max wav data size" hacks that break timing in browsers.
    return (
        b"RIFF" + struct.pack("<I", 36) +   # 36 + data_size, but data_size unknown -> keep minimal
        b"WAVE"
        b"fmt " + struct.pack("<I", 16) +
        struct.pack("<H", 1) +             # PCM
        struct.pack("<H", ch) +
        struct.pack("<I", sr) +
        struct.pack("<I", byte_rate) +
        struct.pack("<H", block_align) +
        struct.pack("<H", bps) +
        b"data" + struct.pack("<I", 0)     # unknown
    )
def _wav_stream(frames: Iterable[AudioFrame]):
    sent_header = False
    for frame in frames:
        if not sent_header:
            yield _wav_header(frame.sample_rate, frame.channels, frame.sample_width)
            sent_header = True
        yield frame.data


@router.post("/tts-stream-fast", tags=["TTS"])
def tts_stream_fast(user_input: TTSRequest, req: Request):
    """
    Fast WAV streaming without ffmpeg, mimicking the legacy low-latency path.
    """
    chat = getattr(req.app.state, "chat", None)
    registry = getattr(req.app.state, "voice_registry", None)
    if chat is None or registry is None:
        raise HTTPException(status_code=500, detail="Chat or voice registry not initialized")

    voice, cfg = registry.get(user_input.language_code().value)
    tts_new = PiperStreamer(
        voice,
        cfg,
        max_chunk_len=85,
        min_break_len=60,
        first_chunk_cb=None,
    )

    if hasattr(chat, "stream"):
        raw_iter = chat.stream(user_input.message, user_input.language_code())
        try:
            first_piece = next(raw_iter)
        except StopIteration:
            raise HTTPException(status_code=502, detail="Stream de texto vazio.")
        except HTTPException as e:
            raise e
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
        text_iter = itertools.chain([first_piece], raw_iter)
    else:
        text_full = chat.stream(user_input.message)
        text_iter = iter([text_full])
    
    def audio_generator() -> Generator[bytes, None, None]:
        header_sent = False
        chunk_iter = buffered(text_iter, language=user_input.language_iso().value)

        expected = None  # (sr, ch, sw)

        def flush_segment(seg: str):
            nonlocal header_sent, expected
            if not seg.strip():
                return
                yield

            processed = seg.strip()

            for sr, ch, sw, pcm in tts_new.stream_pcm(processed):
                # ✅ validate format once and forever
                if expected is None:
                    expected = (sr, ch, sw)
                    logger.info("TTS audio format: sr=%s ch=%s sw=%s", sr, ch, sw)
                else:
                    if (sr, ch, sw) != expected:
                        raise RuntimeError(f"Audio format changed mid-stream: {expected} -> {(sr,ch,sw)}")

                # ✅ validate PCM alignment for s16le
                if sw == 2 and (len(pcm) % 2) != 0:
                    raise RuntimeError(f"PCM chunk not aligned to int16: len={len(pcm)}")

                if not header_sent:
                    yield _wav_header(sr, ch, sw)
                    header_sent = True

                yield pcm

        for chunk in chunk_iter:
            yield from flush_segment(chunk)

        logger.info("TTS audio_generator done | expected=%s", expected)

    # def audio_generator() -> Generator[bytes, None, None]:
    #     header_sent = False
    #     chunk_iter = buffered(text_iter, language=user_input.language_iso().value)

    #     def flush_segment(seg: str):
    #         nonlocal header_sent
    #         if not seg.strip():
    #             return
    #         logger.info("TTS flush_segment | text_len=%d | text=%r", len(seg), seg[:120])
    #         processed = seg.strip()
    #         try:
    #             for sr, ch, sw, pcm in tts_new.stream_pcm(processed):
    #                 if not header_sent:
    #                     yield _wav_header(sr, ch, sw)
    #                     header_sent = True
    #                 yield pcm
    #         except Exception as exc:
    #             logger.error("TTS flush_segment error: %s", exc)
    #             return

    #     # try:
    #     #     for chunk in chunk_iter:
    #     #         yield from flush_segment(chunk)
    #     # finally:
    #     #     logger.info("TTS audio_generator done")

    return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
                # optional: filled after first chunk? (can’t easily unless you buffer)
            },
        )


@router.get("/languages", tags=["Languages"])
def languages_available():
    try:
        return JSONResponse(get_languages_json())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# @router.post("/tts-stream", tags=["TTS"])
# def tts_stream(
#     user_input: TTSRequest,
#     req: Request,
#     format: str | None = Query(None, description="Optional audio format override (aac|m4a|adts|wav)"),
# ):
#     logger.info("tts_stream request start | lang=%s | fmt=%s | msg_chars=%d", user_input.language, format or user_input.format, len(user_input.message or ""))
#     use_case: StreamTTSUseCase = req.app.state.stream_tts
#     # Query param overrides body format to support ?format=m4a usage
#     fmt = format or user_input.format
#     dto = TTSRequestDTO(
#         message=user_input.message,
#         language=user_input.language_code(), 
#         language_iso=user_input.language_iso(),
#         format=AudioFormat.from_str(fmt or "wav"),
#     )

#     frames = use_case.execute(dto)

#     # Force WAV as the default/pattern unless explicitly requested
#     if dto.format in (AudioFormat.AAC, AudioFormat.M4A, AudioFormat.ADTS):
#         if not shutil.which("ffmpeg"):
#             raise HTTPException(
#                 status_code=500,
#                 detail="ffmpeg is required for AAC/M4A streaming. Install ffmpeg or use format=wav.",
#             )
#         # Use ADTS for lowest-latency AAC (mp4 container can buffer)
#         body = aac_stream(frames, container="adts")
#         media_type = "audio/aac"
#     else:
#         body = _wav_stream(frames)
#         media_type = "audio/wav"

#     logger.info("tts_stream streaming start | fmt=%s | lang=%s", media_type, dto.language.value)

#     def instrumented_body():
#         try:
#             yield from body
#             logger.info("tts_stream streaming complete | fmt=%s | lang=%s", media_type, dto.language.value)
#         except Exception as exc:
#             logger.error("tts_stream streaming error | fmt=%s | lang=%s | err=%s", media_type, dto.language.value, exc)
#             raise

#     return StreamingResponse(
#         instrumented_body(),
#        media_type="text/event-stream",
#         headers={
#             "Transfer-Encoding": "chunked",
#             "Cache-Control": "no-store",
#             "Connection": "keep-alive",
#             "Pragma": "no-cache",
#             "X-Accel-Buffering": "no",
#         },
#     )

@router.post("/tts-stream", tags=["TTS"])
def tts_stream(
    user_input: TTSRequest,
    req: Request,
    format: str | None = Query(None, description="aac|m4a|adts|wav"),
):
    use_case: StreamTTSUseCase = req.app.state.stream_tts
    fmt = (format or user_input.format or "wav").lower()

    dto = TTSRequestDTO(
        message=user_input.message,
        language=user_input.language_code(),
        language_iso=user_input.language_iso(),
        format=AudioFormat.from_str(fmt),
    )

    frames = use_case.execute(dto)

    if dto.format == AudioFormat.M4A:
        if not shutil.which("ffmpeg"):
            raise HTTPException(500, "ffmpeg is required for m4a streaming. Use format=wav.")
        # Fragmented MP4 for MSE playback
        body = aac_stream(frames, container="mp4")
        media_type = 'audio/mp4; codecs="mp4a.40.2"'
    elif dto.format in (AudioFormat.AAC, AudioFormat.ADTS):
        if not shutil.which("ffmpeg"):
            raise HTTPException(500, "ffmpeg is required for aac streaming. Use format=wav.")
        # ADTS for plain AAC streaming
        body = aac_stream(frames, container="adts")
        media_type = "audio/aac"
    else:
        body = _wav_stream(frames)
        media_type = "audio/wav"
    
    return StreamingResponse(
        body,
        media_type="audio/wav",
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
            # optional: filled after first chunk? (can’t easily unless you buffer)
        },
    )

    # return StreamingResponse(
    #     body,
    #     media_type=media_type,
    #     headers={
    #         "Cache-Control": "no-store",
    #         "Pragma": "no-cache",
    #         "X-Accel-Buffering": "no",
    #     },
    # )
