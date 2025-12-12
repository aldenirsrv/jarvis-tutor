import os
import io
import re
from typing import Generator, Iterable, Optional
import numpy as np
from piper import PiperVoice, SynthesisConfig
from scipy.io.wavfile import write as write_wav
import time
from typing import Iterable, Generator, Optional, Union

class PiperStreamer:
    """
    Gera áudio TTS em *chunks* usando Piper e o entrega sob demanda
    (ideal para StreamingResponse / WebSocket).

    Exemplo de uso com FastAPI:

        from fastapi import FastAPI
        from fastapi.responses import StreamingResponse

        tts = PiperStreamer()

        app = FastAPI()

        @app.get("/tts-stream")
        def tts_stream(text: str):
            return StreamingResponse(
                tts.stream_wav(text),
                media_type="audio/wav"
            )
    """

    def __init__(
        self,
        model_file: str = "en_US-ryan-medium.onnx",
        model_dir: str = "voices",
        *,
        length_scale: float = 1.2,
        max_chunk_len: int = 220,
        use_cuda: bool = True,
        first_chunk_cb: Optional[callable] = None, 
    ) -> None:
        """
        Parameters
        ----------
        model_file : str
            Nome do arquivo ONNX do modelo.
        model_dir : str
            Pasta onde o modelo está localizado.
        length_scale : float
            Controla a velocidade (1.0 = normal, >1.0 = mais lento).
        max_chunk_len : int
            Nº máximo de caracteres por chunk antes de dividir a frase.
        use_cuda : bool
            Habilita GPU se disponível.
        """
        model_path = os.path.join(model_dir, model_file)
        self.first_chunk_cb = first_chunk_cb
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        # Carrega voz e define configuração de síntese
        self.voice: PiperVoice = PiperVoice.load(model_path, use_cuda=use_cuda)
        # Ajustes para um som mais natural
        self.config = SynthesisConfig(
            length_scale=length_scale,
            volume=1.0,
            noise_scale=0.5,
            noise_w_scale=0.6,
            normalize_audio=True,
        )
        self.max_chunk_len = max_chunk_len
        # Tamanho de chunk menor => primeiro áudio sai mais rápido
        self._wav_chunk_size = 4096

    # --------------------------------------------------------------------- #
    # API pública                                                            #
    # --------------------------------------------------------------------- #
    def stream_wav(self, text: str) -> Generator[bytes, None, None]:
        """
        Gera *bytes* de áudio WAV incrementalmente a partir do texto.

        Pode ser passado direto para StreamingResponse ou WebSocket send().
        """
        for sentence in self._split_text(text):
            for wav_chunk in self._synthesize(sentence):
                yield wav_chunk

    # --------------------------------------------------------------------- #
    # Helpers internos                                                       #
    # --------------------------------------------------------------------- #
    def _split_text(self, text: str) -> Iterable[str]:
        """
        Divide o texto em sentenças respeitando ponto, interrogação
        ou exclamação, e garante que cada pedaço não ultrapasse
        `max_chunk_len` caracteres (para evitar estouro de memória).
        """
        # Primeira quebra: pontuação clássica
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        current = ""

        for sentence in sentences:
            # Se ‘sentence’ ainda é muito grande, force cortes menores
            while len(sentence) > self.max_chunk_len:
                head, sentence = (
                    sentence[: self.max_chunk_len],
                    sentence[self.max_chunk_len :],
                )
                sentences.insert(0, head)  # processar head primeiro
            # Concatena enquanto couber
            if len(current) + len(sentence) <= self.max_chunk_len:
                current += " " + sentence if current else sentence
            else:
                yield current.strip()
                current = sentence
        if current:
            yield current.strip()
    def _synthesize(self, sentence: str) -> Generator[bytes, None, None]:
        """
        Converte uma sentença em áudio e devolve como pedaços WAV.
        Compatível com piper.voice.AudioChunk (retorno padrão do PiperVoice).
        """
        chunks = list(self.voice.synthesize(sentence, self.config))  # retorna AudioChunk
        if not chunks:
            return

        # Usa samples (np.ndarray) e sample_rate dos chunks
        pcm_array = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        sample_rate = chunks[0].sample_rate

        buf = io.BytesIO()
        write_wav(buf, sample_rate, pcm_array)
        buf.seek(0)

        while chunk := buf.read(self._wav_chunk_size):
            yield chunk

    def _pieces(self, text_or_iter: Union[str, Iterable[str]]) -> Iterable[str]:
        if isinstance(text_or_iter, str):
            yield text_or_iter
        else:
            for piece in text_or_iter:
                if piece:
                    yield piece

    def stream_pcm(self, text_or_iter: Union[str, Iterable[str]]
                    ) -> Generator[tuple[int,int,int,bytes], None, None]:
        """
        Stream raw PCM tuples (sr, channels, sample_width, bytes) from text or iterable of text.
        """
        first = True
        t0 = time.perf_counter()

        # Accept a single string OR an iterator of strings
        for piece in self._pieces(text_or_iter):
            # reuse your chunker for each incoming piece
            for sentence in self._split_text(piece):
                # Use PiperVoice directly to keep AudioChunk metadata (pcm, sr, etc.)
                for chunk in self.voice.synthesize(sentence, self.config):
                    pcm = chunk.audio_int16_bytes
                    if not pcm:
                        continue
                    if first and self.first_chunk_cb:
                        self.first_chunk_cb(int((time.perf_counter() - t0) * 1000))
                        first = False
                    yield (chunk.sample_rate, chunk.sample_channels, chunk.sample_width, pcm)