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
from app.streaming import PiperStreamer 

app = FastAPI()
load_dotenv()


class UserInput(BaseModel):
    message: str

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
tts_new = PiperStreamer(
    model_file="en_US-ryan-high.onnx",  # ajuste se necessário
    model_dir="voices",
    length_scale=1.1,                   # 1.0 = normal, >1 = mais lento
    max_chunk_len=80,                   # chunks menores => áudio começa mais rápido
    use_cuda=True,
)

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
def wav_stream(text: str):
    """
    Gera um único fluxo WAV concatenado.
    O primeiro chunk vem com cabeçalho RIFF.
    Nos chunks seguintes descartamos os 44 bytes de cabeçalho
    para evitar múltiplos headers no mesmo arquivo.
    """
    first_chunk = True
    for chunk in tts_new.stream_wav(text):          # cada chunk = WAV completo de 1 frase
        if first_chunk:
            yield chunk                            # mantém RIFF na primeira vez
            first_chunk = False
        else:
            yield chunk[44:]                       # remove cabeçalho nas demais


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
        # audio_bytes = base64.b64decode(audio_b64)

        #       # 3. Salvando o áudio como um arquivo WAV localmente (opcional)
        # audio_file = "output_new.wav"
        # with open(audio_file, "wb") as f:
        #     f.write(audio_bytes)  # Note que salvamos os BYTES aqui (não a string Base64)


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


# @app.post("/tts")
# def chat_endpoint(user_input: UserInput):
#     try:
#         # 1. Gerando o áudio em bytes
#         audio_bytes = tts.run(user_input.message)

#         # 2. Convertendo os bytes para Base64
#         audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
#         audio_bytes = base64.b64decode(audio_b64)

#         # 3. Salvando o áudio como um arquivo WAV localmente (opcional)
#         audio_file = "output.wav"
#         with open(audio_file, "wb") as f:
#             f.write(audio_bytes)  # Note que salvamos os BYTES aqui (não a string Base64)


#         # 4. Retornando a resposta com áudio no formato Base64
#         return {
#             "response": "Aqui vai o texto da resposta",
#             "audio_base64": audio_b64,
#         }

#     except Exception as e:
#         # Tratamento de exceções
#         raise HTTPException(status_code=500, detail=str(e))
# Dummy chat.run simulando resposta gerada — substitua pelo seu chat
def fake_chat_run(text: str) -> str:
    # Aqui você usa seu modelo real de geração de texto
    return (
        "Glad you're interested in Stranger Things! This popular Netflix series revolves around a group of friends in Hawkins, a small town in the Midwest. "
        "They uncover a series of supernatural mysteries when a young boy goes missing. "
        "Throughout the series, the friends confront a shadowy government agency, supernatural creatures, and a mysterious girl with powerful abilities. "
        "There's a lot of adventure, suspense, and some terrifying moments along the way. "
        "It's a thrilling series that will keep you on the edge of your seat, plus it has a great 80s vibe that adds to the overall atmosphere. "
        "Have fun watching and let me know what you think!"
    )
@app.post("/chat-stream")
async def chat_stream_endpoint(user_input: UserInput):
    try:
        # 1. Gera texto
        response_text = chat.run(user_input.message)
        response_processed = add_natural_pauses(response_text)

        # 2. Gera áudio inteiro (para fechar a conexão) e envia em streaming simples
        audio_bytes = tts_piper.run(response_processed)
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/wav",
        )

    except HTTPException as e:
        raise e
    except HfHubHTTPError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tts")
def tts_stream():
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

# 3. Endpoint de streaming ----------------------------------
@app.get("/tts-stream", response_class=StreamingResponse, tags=["TTS"])
async def tts_stream():
# async def tts_stream(text: str = Query(..., min_length=1, max_length=10_000, description="Texto a ser sintetizado")):
    """
    Recebe um texto arbitrário e devolve áudio **WAV** em streaming.

    No frontend, basta fazer um fetch para `/tts-stream?text=...`
    e tocar o stream num objeto `<audio>` ou em **Media Source Extensions**.
    """
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
    try:
        return StreamingResponse(
            tts_new.stream_wav(text),
            media_type="audio/wav",
            headers={
                # Opcional: força download
                # "Content-Disposition": 'attachment; filename="tts.wav"'
            },
        )
    except Exception as exc:
        # Trate erros de forma amigável
        raise HTTPException(status_code=500, detail=str(exc)) from exc
@app.post("/alexa")
async def alexa_endpoint(req: Request):
    body = await req.json()
    logger.info(f"Requisição recebida: {body}")
    req_type = body.get("request", {}).get("type")

    if req_type == "LaunchRequest":
        welcome = "<speak>Hello! I'm Jarvis, your English tutor. You can say things like: <break time='400ms'/> 'Help me with past simple' or 'Let's talk about travel.'</speak>"
        return JSONResponse({
            "version": "1.0",
            "response": {
                "outputSpeech": {"type": "SSML", "ssml": welcome},
                "shouldEndSession": False
            }
        })

    if req_type == "IntentRequest":
        message = body.get("request", {}).get("intent", {}).get("slots", {}).get("message", {}).get("value", "")
       
        if not message:
            fallback = "<speak>I'm here to help you practice English. Try saying something like, <break time='400ms'/> 'Help me with vocabulary' or 'Talk to me about travel.'</speak>"
            return JSONResponse({
                "version": "1.0",
                "response": {
                    "outputSpeech": {"type": "SSML", "ssml": fallback},
                    "shouldEndSession": False
                }
            })

        try:
            response = chat.run(message)
            ssml_output = clean_response_for_alexa(response)
        except Exception as e:
            print(f"[ERROR] {e}")
            ssml_output = "<speak>Sorry, I had trouble processing your request. Please try again.</speak>"

        return JSONResponse({
            "version": "1.0",
            "response": {
                "outputSpeech": {
                    "type": "SSML",
                    "ssml": ssml_output
                },
                "shouldEndSession": False
            }
        })

    # Catch-all fallback
    return JSONResponse({
        "version": "1.0",
        "response": {
            "outputSpeech": {
                "type": "SSML",
                "ssml": "<speak>Sorry, I couldn't understand your request.</speak>"
            },
            "shouldEndSession": False
        }
    })
@app.get("/tts-stream-ogg")
def tts_stream_ogg(text: str = Query(..., min_length=1)):
    # Gera WAV completo em memória
    wav_buffer = io.BytesIO()
    for chunk in tts_new.stream_wav(text):
        wav_buffer.write(chunk)
    wav_buffer.seek(0)

    # Cria processo FFmpeg para converter WAV para OGG Opus via stdin/stdout
    ffmpeg_cmd = [
        "ffmpeg",
        "-i", "pipe:0",          # entrada stdin
        "-c:a", "libopus",       # codec Opus
        "-f", "ogg",             # formato OGG
        "pipe:1",                # saída stdout
    ]

    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def stream():
        try:
            # Envia WAV para o FFmpeg stdin
            process.stdin.write(wav_buffer.read())
            process.stdin.close()

            # Lê o OGG convertido em chunks e streama
            while True:
                data = process.stdout.read(4096)
                if not data:
                    break
                yield data
        finally:
            process.stdout.close()
            process.wait()

    return StreamingResponse(stream(), media_type="audio/ogg")
# if __name__ == "__main__":

#     tts = PiperTTSStream()
#     text = """No worries, Aldenir! I simply suggested two movies for you to watch. 
#     The first one is Inception, a 2010 sci-fi thriller directed by Christopher Nolan. 
#     Alternatively, I suggested Ghostbusters, a hilarious supernatural comedy from 1984."""

#     output_path = "test_output.wav"
#     with open(output_path, "wb") as f:
#         f.write(tts.run(text))

#     print(f"Áudio salvo em: {os.path.abspath(output_path)}")
