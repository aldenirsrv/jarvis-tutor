import os
import io
import re
from typing import Generator, Iterable, Optional
import time
from typing import Iterable, Generator, Optional, Union

# piper_streamer.py
class PiperStreamer:
    def __init__(
        self,
        voice,
        synth_cfg,                          # PiperConfig (or kwargs in older code)
        *,
        max_chunk_len: int = 160,
        min_break_len: int = 60,
        first_chunk_cb: Optional[callable] = None,   # <-- add
    ):
        self.voice = voice
        self.synth_cfg = synth_cfg
        self.max_chunk_len = max_chunk_len
        self.min_break_len = min_break_len
        self.first_chunk_cb = first_chunk_cb

    def _split_text(self, text: str) -> Iterable[str]:
        text = text.strip()
        if not text: return []
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for s in sentences:
            s = s.strip()
            if not s: continue
            buf, n = [], 0
            for ch in s:
                buf.append(ch); n += 1
                if ch in ".?!;," and n >= self.min_break_len:
                    yield "".join(buf).strip(); buf=[]; n=0
                elif n >= self.max_chunk_len:
                    yield "".join(buf).strip(); buf=[]; n=0
            if buf: yield "".join(buf).strip()

    def _synthesize(self, sentence: str):
        # compatible with both Piper APIs
        if isinstance(self.synth_cfg, dict):
            return self.voice.synthesize(sentence, **self.synth_cfg)
        return self.voice.synthesize(sentence, self.synth_cfg)

    def stream_pcm(self, text_or_iter: Union[str, Iterable[str]]) -> Generator[tuple[int,int,int,bytes], None, None]:
        """Accept a single string or an iterator of strings and stream PCM chunks."""
        first = True
        t0 = time.perf_counter()

        # Normalize to an iterator of sentences
        if isinstance(text_or_iter, str):
            sentence_iter = self._split_text(text_or_iter)
        else:
            def _sentence_iter():
                for piece in text_or_iter:
                    if piece:
                        yield from self._split_text(piece)
            sentence_iter = _sentence_iter()

        for sentence in sentence_iter:
            for chunk in self._synthesize(sentence):
                pcm = chunk.audio_int16_bytes
                if not pcm:
                    continue
                if first and self.first_chunk_cb:
                    self.first_chunk_cb(int((time.perf_counter() - t0) * 1000))
                    first = False
                yield (chunk.sample_rate, chunk.sample_channels, chunk.sample_width, pcm)
