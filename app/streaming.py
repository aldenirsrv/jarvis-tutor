import os
import io
import re
from typing import Generator, Iterable
import numpy as np
from piper import PiperVoice, SynthesisConfig
from scipy.io.wavfile import write as write_wav


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
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")

        # Carrega voz e define configuração de síntese
        self.voice: PiperVoice = PiperVoice.load(model_path, use_cuda=use_cuda)
        self.config = SynthesisConfig(length_scale=length_scale)
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


#   first_chunk = next(audio_chunks)
#         print(dir(first_chunk))['__annotations__', '__class__', '__dataclass_fields__', '__dataclass_params__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__firstlineno__', '__format__', '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__match_args__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__replace__', '__repr__', '__setattr__', '__sizeof__', '__static_attributes__', '__str__', '__subclasshook__', '__weakref__', '_audio_int16_array', '_audio_int16_bytes', 'audio_float_array', 'audio_int16_array', 'audio_int16_bytes', 'sample_channels', 'sample_rate', 'sample_width']
