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


@router.get("/languages", tags=["Languages"])
def languages_available():
    try:
        return JSONResponse(get_languages_json())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
            # optional: filled after first chunk? (can’t easily unless you buffer)
        },
    )

