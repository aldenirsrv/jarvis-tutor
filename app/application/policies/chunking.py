from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, List, Optional

import nltk
from nltk.tokenize.punkt import PunktSentenceTokenizer

logger = logging.getLogger(__name__)

# sentence end (handles quotes/brackets at end)
_STRONG_END = re.compile(r'[.!?…]+(?:["\')\]]+)?\s*$')
# find any sentence end inside text (for streaming cuts)
_STRONG_END_ANY = re.compile(r'[.!?…]+(?:["\')\]]+)?\s+')
# clause boundary for emergency splits
_CLAUSE = re.compile(r'[,;:]+(?=\s|$)')

Lang = Literal["en", "pt", "es", "fr"]

_PUNKT_LANG = {
    "en": "english",
    "pt": "portuguese",
    "es": "spanish",
    "fr": "french",
}

def _load_punkt(lang: Lang) -> PunktSentenceTokenizer:
    # Ensure punkt is available (do this once at app startup ideally)
    # nltk.download("punkt", quiet=True)  # <-- better in startup, not per-request
    return nltk.data.load(f"tokenizers/punkt/{_PUNKT_LANG[lang]}.pickle")

def _split_long_sentence_coherent(s: str, max_chars: int, min_piece: int = 25) -> List[str]:
    """
    Only used when a single sentence is WAY too long. Prefer clause splits, avoid tiny pieces.
    """
    s = s.strip()
    if len(s) <= max_chars:
        return [s]

    out: List[str] = []
    buf = s
    while len(buf) > max_chars:
        best: Optional[int] = None
        for m in _CLAUSE.finditer(buf):
            if m.end() <= max_chars:
                best = m.end()
            else:
                break

        cut = best
        if cut is None or cut < min_piece:
            cut = buf.rfind(" ", min_piece, max_chars + 1)
            if cut == -1:
                cut = max_chars

        piece = buf[:cut].strip()
        buf = buf[cut:].lstrip()

        # avoid tiny pieces: if piece is tiny, merge it back by cutting later next iteration
        if out and len(piece) < min_piece:
            out[-1] = (out[-1] + " " + piece).strip()
        else:
            out.append(piece)

    if buf:
        if out and len(buf) < min_piece:
            out[-1] = (out[-1] + " " + buf).strip()
        else:
            out.append(buf.strip())
    return out

@dataclass
class StreamingSentenceChunker:
    language: Lang = "en"
    min_chunk_chars: int = 55
    max_chunk_chars: int = 80

    # streaming controls
    grace_chars: int = 12          # allow small overflow to avoid micro-tails
    hard_max_buffer: int = 160     # only force-cut if buffer gets huge
    min_sentence_tail: int = 12    # attach tiny "alguém." to previous chunk

    def __post_init__(self):
        self.tokenizer = _load_punkt(self.language)
        self.buf = ""

    def _segment(self, text: str) -> List[str]:
        spans = list(self.tokenizer.span_tokenize(text))
        return [text[a:b] for a, b in spans if text[a:b].strip()]

    def feed(self, part: str) -> Iterator[str]:
        if not part:
            return
        self.buf += part

        # Wait until we have enough text, and either:
        #  - we see a strong sentence end at the end, or
        #  - buffer is getting dangerously large (hard_max_buffer)
        if len(self.buf) < self.min_chunk_chars and len(self.buf) < self.hard_max_buffer:
            return

        if not _STRONG_END.search(self.buf) and len(self.buf) < self.hard_max_buffer:
            # no clear end yet; keep buffering to avoid mid-sentence cuts
            return

        # Segment current buffer
        sentences = self._segment(self.buf)

        # If buffer doesn't end with punctuation, keep last sentence as remainder
        remainder = ""
        if sentences and not _STRONG_END.search(self.buf):
            remainder = sentences[-1]
            sentences = sentences[:-1]

        self.buf = remainder.strip()

        yield from self._pack(sentences, final=False)

        # If buffer exceeded hard limit and we *still* have no punctuation,
        # do an emergency coherent cut (rare)
        if len(self.buf) >= self.hard_max_buffer and not _STRONG_END_ANY.search(self.buf):
            cut = self.buf.rfind(" ", 40, self.max_chunk_chars + 1)
            if cut == -1:
                cut = self.max_chunk_chars
            emergency = self.buf[:cut].strip()
            self.buf = self.buf[cut:].lstrip()
            if emergency:
                yield emergency

    def flush(self) -> Iterator[str]:
        tail = self.buf.strip()
        self.buf = ""
        if not tail:
            return
        sentences = self._segment(tail)
        yield from self._pack(sentences, final=True)

    def _pack(self, sentences: List[str], final: bool) -> Iterator[str]:
        chunk = ""

        def emit(c: str, reason: str):
            logger.info("chunker flush | reason=%s | chars=%d | preview=%s", reason, len(c), c[:80])
            return c

        for s in sentences:
            s = s.strip()
            # only split if sentence itself is too long
            parts = _split_long_sentence_coherent(s, self.max_chunk_chars, min_piece=25)

            for p in parts:
                p = p.strip()
                if not p:
                    continue

                # attach tiny tails to previous chunk if possible
                if chunk and len(p) <= self.min_sentence_tail and len(chunk) <= self.max_chunk_chars + self.grace_chars:
                    chunk = (chunk + " " + p).strip()
                    continue

                candidate = (chunk + " " + p).strip() if chunk else p

                if len(candidate) <= self.max_chunk_chars:
                    chunk = candidate
                    continue

                # allow small overflow to avoid chopping coherence
                if len(candidate) <= self.max_chunk_chars + self.grace_chars:
                    chunk = candidate
                    continue

                if chunk:
                    yield emit(chunk, "packed")
                    chunk = p
                else:
                    yield emit(p, "hard")

        if chunk:
            if final or len(chunk) >= self.min_chunk_chars:
                yield emit(chunk, "final/packed")
            else:
                # In streaming, it's usually better to keep this in buffer,
                # but for TTS you may still prefer emitting. Here we emit.
                yield emit(chunk, "small")
def buffered(iter_text: Iterable[str], *, language: Lang = "en",
            min_chars=55, max_chars=80) -> Iterator[str]:
    ch = StreamingSentenceChunker(language=language,
                                 min_chunk_chars=min_chars,
                                 max_chunk_chars=max_chars)
    for part in iter_text:
        yield from ch.feed(part)
    yield from ch.flush()
