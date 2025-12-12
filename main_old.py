from fastapi import FastAPI, Request, HTTPException, Query
from huggingface_hub.utils import HfHubHTTPError
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from app.model import HuggingFaceChat, OpenAIChat
from app.text_to_speech import EdgeTTS
from app.tts_piper import PiperTTS, PiperTTSStream
from app.memory import SQLiteMemory
from fastapi.middleware.cors import CORSMiddleware
from app.pre_processor import add_natural_pauses
from app.languages import get_voice, get_scale, get_languages_json, LangInfo
from pydantic import BaseModel
import logging
from dotenv import load_dotenv
import os
import subprocess
import io
import re
import html
import base64
from fastapi.responses import StreamingResponse
from app.streaming import PiperStreamer, PiperStreamerV2
from typing import Generator
import struct
import itertools
from typing import Dict, Iterable
from app.voices import VoiceRegistry
from contextlib import asynccontextmanager
load_dotenv()
import subprocess, shutil, struct
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
tts = EdgeTTS(voice="en-US-GuyNeural")
tts_piper = PiperTTS(length_scale=1.25)
tts_instance = PiperTTSStream()
# 1. Instancia o gerador de áudio ---------------------------

def clean_response_for_alexa(text: str, max_sentences=4) -> str:
    try: 
        # Escape qualquer caractere especial que quebre o SSML (principalmente &, <, >)
        text = html.escape(text)

        # Remove quebras de linha e hifens de listas
        text = text.replace("\n", " ")
        text = re.sub(r"[-•*]\s*", "", text)  # remove bullets/hifens

        # Remove múltiplos espaços
        text = re.sub(r"\s+", " ", text).strip()

        # Remove números no início das frases (ex: "1. Example")
        text = re.sub(r"\d+\.\s*", "", text)

        # Divide em sentenças (limitadas a X)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = sentences[:max_sentences]

        # Constrói SSML
        ssml = "<speak>"
        for sentence in sentences:
            if sentence:  # ignora strings vazias
                ssml += f"{sentence.strip()} <break time='350ms'/> "
        ssml += "</speak>"

        return ssml
    except HfHubHTTPError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-stream-text")
async def chat_stream_text(user_input: UserInput):
    """
    Streaming de texto (OpenAI backend). Útil para reduzir latência de resposta.
    """
    if not isinstance(chat, OpenAIChat):
        raise HTTPException(status_code=400, detail="Streaming de texto disponível apenas com CHAT_BACKEND=openai.")
    try:
        return StreamingResponse(
            chat.stream(user_input.message),
            media_type="text/plain",
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_endpoint(user_input: UserInput):
    try:
        response = chat.run(user_input.message)
        response_processed = add_natural_pauses(response)
        audio_bytes = tts_piper.run(response_processed)
        
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        return {
                    "response": response_processed,
                    "audio_base64": audio_b64
                }
    except HTTPException as e:
        # Propaga HTTPException de forma transparente (ex: erros de token/modelo)
        raise e
    except HfHubHTTPError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat-stream")
async def chat_stream_endpoint(user_input: UserInput):
    """
    Streaming de áudio com início rápido (~<1s):
    - Usa chat stream (se disponível) para ir recebendo texto
    - Sintetiza e envia WAV em pequenos pedaços conforme o texto chega
    """
    MAX_SEG_LEN = 85

    tts_new = PiperStreamer(
        model_file=get_voice(user_input.language),
        model_dir="voices",
        length_scale=get_scale(user_input.language),
        max_chunk_len=MAX_SEG_LEN,
        use_cuda=True,
    )

    try:
        # 1) Fonte de texto (stream se existir)
        if hasattr(chat, "stream"):
            raw_iter = chat.stream(user_input.message, user_input.language)
            try:
                first_piece = next(raw_iter)
            except StopIteration:
                raise HTTPException(status_code=502, detail="Stream de texto vazio.")
            except HTTPException as e:
                raise e
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc))
            # chain primeiro pedaço + restante sem materializar tudo
            text_iter = itertools.chain([first_piece], raw_iter)
        else:
            text_full = chat.stream(user_input.message)
            text_iter = iter([text_full])

        def audio_generator() -> Generator[bytes, None, None]:
            buffer = ""
            header_sent = False

            def make_stream_header(sr: int, ch: int, sw: int) -> bytes:
                byte_rate = sr * ch * sw
                block_align = ch * sw
                bits_per_sample = sw * 8
                data_size = 0xFFFFFFFF
                riff_size = 0xFFFFFFFF
                return (
                    b"RIFF"
                    + struct.pack("<I", riff_size)
                    + b"WAVEfmt "
                    + struct.pack("<I", 16)  # fmt chunk size
                    + struct.pack("<H", 1)   # PCM
                    + struct.pack("<H", ch)
                    + struct.pack("<I", sr)
                    + struct.pack("<I", byte_rate)
                    + struct.pack("<H", block_align)
                    + struct.pack("<H", bits_per_sample)
                    + b"data"
                    + struct.pack("<I", data_size)
                )

            def flush_segment(seg: str):
                nonlocal header_sent
                if not seg.strip():
                    return
                logger.info("TTS flush_segment | text_len=%d | text=%r", len(seg), seg[:120])
                processed = add_natural_pauses(seg.strip())
                logger.info("add_natural_pauses processed | text_len=%d", len(processed))
                # Usa PCM por sentença; envia um único cabeçalho de stream
                try:
                    for sr, ch, sw, pcm in tts_new.stream_pcm(processed):
                        if not header_sent:
                            yield make_stream_header(sr, ch, sw)
                            header_sent = True
                        yield pcm
                except Exception as exc:
                    logger.error("TTS flush_segment error: %s", exc)
                    return

            try:
                for piece in text_iter:
                    buffer += piece
                    # print(f"buffer: {len(buffer)}: {buffer}")

                    while True:
                        # If buffer is short and has no strong punctuation, wait for more text
                        if len(buffer) <= MAX_SEG_LEN and not any(ch in buffer for ch in ".!?"):
                            break

                        split_pos = -1
                        forced_split = False

                        # 1) Try to split on punctuation within MAX_SEG_LEN
                        #    (we only care about the portion up to MAX_SEG_LEN)
                        search_window = min(len(buffer), MAX_SEG_LEN)
                        for p in ".!?":
                            idx = buffer.rfind(p, 0, search_window)
                            if idx > split_pos:
                                split_pos = idx

                        if split_pos != -1:
                            # include the punctuation character
                            end_idx = split_pos + 1
                            segment = buffer[:end_idx].strip()
                        elif len(buffer) > MAX_SEG_LEN:
                            # 2) No punctuation: force split near a space before MAX_SEG_LEN
                            forced_split = True
                            space_idx = buffer.rfind(" ", 0, MAX_SEG_LEN)

                            if space_idx == -1:
                                # no space, hard cut so we respect MAX_SEG_LEN - 3 for "..."
                                end_idx = MAX_SEG_LEN - 3
                                raw = buffer[:end_idx].rstrip()
                            else:
                                end_idx = space_idx
                                raw = buffer[:end_idx].rstrip()
                                # still enforce max length with ellipsis
                                if len(raw) > MAX_SEG_LEN - 3:
                                    raw = raw[: MAX_SEG_LEN - 3].rstrip()

                            segment = raw + "..."
                        else:
                            # buffer <= MAX_SEG_LEN and no punctuation: nothing to flush yet
                            break

                        # Safety: enforce final cap (defensive)
                        if len(segment) > MAX_SEG_LEN:
                            segment = segment[: MAX_SEG_LEN]

                        # Advance buffer
                        buffer = buffer[end_idx:].lstrip()

                        logger.info(
                            "TTS flush_from_loop | segment_len=%d | remaining_len=%d",
                            len(segment),
                            len(buffer),
                        )
                        yield from flush_segment(segment)


                # Finaliza resto
                # if buffer.strip():
                #     logger.info("TTS flush_final | len=%d | text=%r", len(buffer), buffer[:120])
                #     yield from flush_segment(buffer)
                if buffer.strip():
                    while len(buffer) > MAX_SEG_LEN:
                        space_idx = buffer.rfind(" ", 0, MAX_SEG_LEN)
                        forced_split = True
                        if space_idx == -1:
                            end_idx = MAX_SEG_LEN - 3
                            raw = buffer[:end_idx].rstrip()
                        else:
                            end_idx = space_idx
                            raw = buffer[:end_idx].rstrip()
                            if len(raw) > MAX_SEG_LEN - 3:
                                raw = raw[: MAX_SEG_LEN - 3].rstrip()

                        segment = raw + "..."
                        if len(segment) > MAX_SEG_LEN:
                            segment = segment[: MAX_SEG_LEN]

                        buffer = buffer[end_idx:].lstrip()
                        logger.info(
                            "TTS flush_final_loop | segment_len=%d | remaining_len=%d",
                            len(segment),
                            len(buffer),
                        )
                        yield from flush_segment(segment)

                    # Último pedaço (já <= MAX_SEG_LEN)
                    if buffer.strip():
                        logger.info(
                            "TTS flush_final | len=%d | text=%r", len(buffer), buffer[:120]
                        )
                        yield from flush_segment(buffer.strip())
            finally:
                logger.info("TTS audio_generator done")

        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav",
            headers={"Transfer-Encoding": "chunked"},
        )

    except HTTPException as e:
        raise e
    except HfHubHTTPError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts", response_class=StreamingResponse, tags=["TTS"])
def tts_stream():
    try:
        text = (
            "Enhance the chatbot experience by introducing user-specific personalization features. "
            "This epic focuses on enabling the AI-powered chatbot to remember past interactions, "
            "tailor responses based on user preferences or history, and recognize returning users for "
            "a more seamless and human-like experience. By storing chat history and leveraging contextual data, "
            "the system will deliver smarter, more relevant, and consistent support over time."
            "Glad you're interested in Stranger Things! This popular Netflix series revolves around a group of friends in Hawkins, a small town in the Midwest. "
            "They uncover a series of supernatural mysteries when a young boy goes missing. "
            "Throughout the series, the friends confront a shadowy government agency, supernatural creatures, and a mysterious girl with powerful abilities. "
            "There's a lot of adventure, suspense, and some terrifying moments along the way. "
            "It's a thrilling series that will keep you on the edge of your seat, plus it has a great 80s vibe that adds to the overall atmosphere. "
            "Have fun watching and let me know what you think!"
        )
        return StreamingResponse(
            tts_instance.stream_wav(text),
            media_type="audio/wav"
        )
    except HTTPException as e:
        # Propaga HTTPException de forma transparente (ex: erros de token/modelo)
        raise e
    except HfHubHTTPError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/languages",  response_class=Dict[str, LangInfo], tags=["Languages"])
def languages_available():
        try:
            return JSONResponse(get_languages_json())  # exact pass-through()
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))