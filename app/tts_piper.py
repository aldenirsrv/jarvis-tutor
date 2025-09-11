import io, os, wave
from piper import PiperVoice, SynthesisConfig
from typing import Generator, Iterable
from sentence_splitter import SentenceSplitter


# MODEL_NAME = "en_US-ryan-high.onnx"

class PiperTTS:
    def __init__(self, length_scale):
        model_dir = "voices"
        model_file = "en_US-ryan-medium.onnx"
        model_path = os.path.join(model_dir, model_file)
       
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        # baixa .onnx + .json
        self.voice = PiperVoice.load(model_path, use_cuda=True)

        self.config = SynthesisConfig(
            length_scale=length_scale,  # controla velocidade
            volume=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            normalize_audio=True,
        )

    def run(self, text: str) -> bytes:
        chunks = list(self.voice.synthesize(text, self.config))        # AudioChunk[]
        print(self.config)
        if not chunks:
            return b""

        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        sr  = chunks[0].sample_rate
        ch  = chunks[0].sample_channels
        sw  = chunks[0].sample_width                    # bytes por amostra (2)

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(sw)
            wf.setframerate(sr)
            wf.writeframes(pcm)
        return buf.getvalue()             

class PiperTTSStream:
    def __init__(self, length_scale=1.2, use_cuda=True):
        model_dir = "voices"
        model_file = "en_US-ryan-high.onnx"
        model_path = os.path.join(model_dir, model_file)

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        self.voice = PiperVoice.load(model_path, use_cuda=use_cuda)

        self.config = SynthesisConfig(
            length_scale=length_scale,
            volume=1.0,
            noise_scale=0.667,
            noise_w_scale=0.8,
            normalize_audio=True,
        )

        # self.splitter = SentenceSplitter(language="en")

    def synthesize_wav_bytes(self, text: str) -> bytes:
        """
        Gera o áudio WAV completo do texto, com cabeçalho WAV correto.
        """
        chunks = list(self.voice.synthesize(text, self.config))
        if not chunks:
            return b""

        pcm = b"".join(c.audio_int16_bytes for c in chunks)
        sr = chunks[0].sample_rate
        ch = chunks[0].sample_channels
        sw = chunks[0].sample_width

        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(ch)
            wf.setsampwidth(sw)
            wf.setframerate(sr)
            wf.writeframes(pcm)

        return buf.getvalue()

    def stream_wav(self, text: str, chunk_size: int = 4096) -> Generator[bytes, None, None]:
        """
        Gera o WAV completo em bytes e faz streaming em pedaços (chunks).
        """
        wav_bytes = self.synthesize_wav_bytes(text)
        buf = io.BytesIO(wav_bytes)

        while True:
            chunk = buf.read(chunk_size)
            if not chunk:
                break
            yield chunk
    # def __init__(self):
    #     model_dir = "voices"
    #     model_file = "en_US-ryan-high.onnx"
    #     model_path = os.path.join(model_dir, model_file)
       
    #     if not os.path.isfile(model_path):
    #         raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

    #     # Carrega o modelo (piper busca automaticamente o .json ao lado do .onnx)
    #     self.voice = PiperVoice.load(model_path)

    # def _ensure_files(self):
    #     for fname in (self.model_name, self.json_name):
    #         if not os.path.exists(fname) or os.path.getsize(fname) == 0:
    #             self._download(fname)

    # def run(self, text: str) -> bytes:
    #     audio_chunks = self.voice.synthesize(text)
    #     audio_bytes = b"".join(chunk.audio_int16_bytes for chunk in audio_chunks)
    #     return audio_bytes