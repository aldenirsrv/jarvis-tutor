import os
from pydantic import BaseModel
from typing import Dict, Iterable
import subprocess, shutil, struct
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from app.model import HuggingFaceChat, OpenAIChat
from app.memory import SQLiteMemory
from app.languages import get_languages_json, LangInfo
from app.streaming import PiperStreamer
from app.voices import VoiceRegistry

import logging
from dotenv import load_dotenv



from contextlib import asynccontextmanager
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = VoiceRegistry()
    registry.load_all()
    app.state.voice_registry = registry
    yield

app = FastAPI(lifespan=lifespan)

class UserInput(BaseModel):
    message: str
    language: str

app.add_middleware(
    CORSMiddleware,
     allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


memory = SQLiteMemory("conversations.db")
hf_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Seleciona backend de chat via variável de ambiente
chat_backend = os.getenv("CHAT_BACKEND", "huggingface").lower()
if chat_backend == "openai":
    chat = OpenAIChat(memory=memory)
else:
    chat = HuggingFaceChat(memory=memory, hf_token=hf_token)

# 1. Instancia o gerador de áudio ---------------------------

@app.get("/languages",  response_class=Dict[str, LangInfo], tags=["Languages"])
def languages_available():
        try:
            return JSONResponse(get_languages_json())  # exact pass-through()
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


def _wav_stream(text_or_iter, streamer: PiperStreamer):
    sent_header = False
    for sr, ch, sw, pcm in streamer.stream_pcm(text_or_iter):
        if not sent_header:
            yield _wav_header(sr, ch, sw); sent_header = True
        yield pcm

def _aac_stream(text_or_iter, streamer: PiperStreamer):
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not found (needed for AAC streaming)")
    sr, ch, sw = 16000, 1, 2
    proc = subprocess.Popen(
        ["ffmpeg","-loglevel","error",
         "-f","s16le","-ar",str(sr),"-ac",str(ch),"-i","pipe:0",
         "-c:a","aac","-b:a","64k","-f","adts","pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        for _sr, _ch, _sw, pcm in streamer.stream_pcm(text_or_iter):
            proc.stdin.write(pcm)
            proc.stdin.flush()
            out = proc.stdout.read1(4096)
            if out:
                yield out
        proc.stdin.close()
        while True:
            out = proc.stdout.read(4096)
            if not out: break
            yield out
    finally:
        proc.terminate()
        
def _wav_header(sr:int, ch:int, sw:int)->bytes:
    byte_rate = sr*ch*sw; block_align = ch*sw; bps = sw*8
    data_size = 0xFFFFFFFF; riff_size = 0xFFFFFFFF
    return (b"RIFF" + struct.pack("<I", riff_size) + b"WAVEfmt " +
            struct.pack("<I",16) + struct.pack("<H",1) + struct.pack("<H",ch) +
            struct.pack("<I",sr) + struct.pack("<I",byte_rate) +
            struct.pack("<H",block_align) + struct.pack("<H",bps) +
            b"data" + struct.pack("<I",data_size))


def buffered(iter_text: Iterable[str], punct=(".","!","?"), min_chars=40, max_chars=160):
    buf = []
    total = 0
    for part in iter_text:
        buf.append(part); total += len(part)
        if total >= min_chars and any(p in part for p in punct):
            out = "".join(buf); buf.clear(); total = 0
            yield out
        elif total >= max_chars:
            out = "".join(buf); buf.clear(); total = 0
            yield out
    if buf:
        yield "".join(buf)

@app.post("/tts-stream")
def tts_stream(user_input: UserInput, req: Request):
    reg: VoiceRegistry = req.app.state.voice_registry
    voice, cfg = reg.get(user_input.language)
    streamer = PiperStreamer(voice, cfg, first_chunk_cb=lambda ms: print("TTS_FIRST_CHUNK(ms)=", ms))

    raw_iter = chat.stream(user_input.message, user_input.language)
    text_iter = buffered(raw_iter)  # iterator of small strings

    fmt = "wav"  # or read from user_input.format
    if fmt in ("m4a","aac","adts"):
        return StreamingResponse(_aac_stream(text_iter, streamer), media_type="audio/aac")
    else:
        return StreamingResponse(_wav_stream(text_iter, streamer), media_type="audio/wav")