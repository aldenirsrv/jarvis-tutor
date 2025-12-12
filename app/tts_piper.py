import io, os, wave
from piper import PiperVoice, SynthesisConfig
from typing import Generator, Iterable
from sentence_splitter import SentenceSplitter


# MODEL_NAME = "en_US-ryan-high.onnx"

class PiperTTS:
    def __init__(self, length_scale):
        model_dir = "voices"
        model_file = "en_US-ryan-high.onnx"
        model_path = os.path.join(model_dir, model_file)
       
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        # baixa .onnx + .json
        self.voice = PiperVoice.load(model_path, use_cuda=True)

        self.config = SynthesisConfig(
            # Ajustes mais naturais: velocidade levemente abaixo e menos ruído
            length_scale=length_scale,  # controla velocidade
            volume=1.0,
            noise_scale=0.5,
            noise_w_scale=0.6,
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
    def __init__(self, length_scale=0.9, use_cuda=True):
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
        Generate the full audio WAV of the text with correct WAV header.
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
        Generate the full WAV in bytes and make the straming in chuncks.
        """
        wav_bytes = self.synthesize_wav_bytes(text)
        buf = io.BytesIO(wav_bytes)

        while True:
            chunk = buf.read(chunk_size)
            if not chunk:
                break
            yield chunk