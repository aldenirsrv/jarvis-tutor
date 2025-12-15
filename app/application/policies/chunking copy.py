from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, Literal, List

import pysbd
# ISO code. Available codes are : {'kk', 'es', 'it', 'hi', 'ru', 'ur', 'da', 'fa', 'ja', 'hy', 'de', 'en', 'ar', 'my', 'sk', 'zh', 'bg', 'pl', 'fr', 'el', 'nl', 'mr', 'am'}

logger = logging.getLogger(__name__)

# Strong “end of sentence” markers (helps decide if last sentence is complete)
_STRONG_END = re.compile(r'[.!?…]+(?:["\')\]]+)?\s*$')

# For splitting a too-long single sentence
_CLAUSE = re.compile(r'[,;:]+(?=\s|$)')

Lang = Literal["en", "pt", "fr", "es"]

def _split_long_sentence(s: str, max_chars: int) -> List[str]:
    s = s.strip()
    if len(s) <= max_chars:
        return [s]

    out: List[str] = []
    buf = s
    while len(buf) > max_chars:
        # prefer clause boundary
        best = None
        for m in _CLAUSE.finditer(buf):
            if m.end() <= max_chars:
                best = m.end()
            else:
                break
        cut = best if best and best >= 10 else None

        if cut is None:
            # fallback: last space
            cut = buf.rfind(" ", 10, max_chars + 1)
            if cut == -1:
                cut = max_chars

        out.append(buf[:cut].strip())
        buf = buf[cut:].lstrip()

    if buf:
        out.append(buf.strip())
    return out

@dataclass
class StreamingSentenceChunker:
    language: Lang = "en"
    min_chunk_chars: int = 35
    max_chunk_chars: int = 55
    grace_chars: int = 6               # wait a bit beyond max for punctuation to arrive
    min_short_sentence: int = 5        # allow short greeting/questions
    split_greetings: bool = True       # "Claro, Aldenir!" as its own chunk

    def __post_init__(self):
        self.seg = pysbd.Segmenter(language=self.language, clean=False)
        self.buf = ""

    def feed(self, part: str) -> Iterator[str]:
        if not part:
            return
        self.buf += part

        # Don’t force-cut at max immediately; give punctuation time to arrive
        if len(self.buf) < self.min_chunk_chars:
            return
        if len(self.buf) < self.max_chunk_chars + self.grace_chars and not _STRONG_END.search(self.buf):
            # still accumulating and no clear end yet
            return

        # Segment buffer into sentences (multilingual)
        sentences = self.seg.segment(self.buf)

        # Keep last sentence un-emitted if it doesn’t look complete yet
        complete: List[str] = []
        remainder = ""
        if sentences:
            if _STRONG_END.search(self.buf):
                complete = [s for s in sentences if s.strip()]
            else:
                complete = [s for s in sentences[:-1] if s.strip()]
                remainder = sentences[-1]  # keep partial

        # Update buffer to only the remainder
        self.buf = remainder.strip() if remainder else ""

        # Now pack complete sentences into chunks <= max_chunk_chars
        yield from self._pack_sentences(complete)

    def flush(self) -> Iterator[str]:
        tail = self.buf.strip()
        self.buf = ""
        if not tail:
            return
        # Segment whatever remains and emit
        sentences = [s for s in self.seg.segment(tail) if s.strip()]
        yield from self._pack_sentences(sentences, final=True)

    def _pack_sentences(self, sentences: List[str], final: bool = False) -> Iterator[str]:
        chunk = ""

        for s in sentences:
            s = s.strip()

            # If sentence itself is too long, split it coherently
            parts = _split_long_sentence(s, self.max_chunk_chars)

            for p in parts:
                # Optional: split greeting/question/exclamation into its own chunk
                if (
                    self.split_greetings
                    and len(p) >= self.min_short_sentence
                    and p[-1] in ("!", "?")
                    and chunk == ""
                ):
                    logger.info("chunker flush | reason=sentence(short) | chars=%d | preview=%s", len(p), p[:80])
                    yield p
                    continue

                # Try to add to current chunk
                candidate = (chunk + " " + p).strip() if chunk else p

                if len(candidate) <= self.max_chunk_chars:
                    chunk = candidate
                    continue

                # Would exceed max: flush current chunk if it’s not empty
                if chunk:
                    logger.info("chunker flush | reason=packed | chars=%d | preview=%s", len(chunk), chunk[:80])
                    yield chunk
                    chunk = ""

                # If p alone still too big (shouldn’t, due to _split_long_sentence), emit it
                if len(p) > self.max_chunk_chars:
                    logger.info("chunker flush | reason=hard | chars=%d | preview=%s", len(p), p[:80])
                    yield p
                else:
                    chunk = p

        if chunk:
            # If it’s the final flush, emit even if smaller than min
            if final or len(chunk) >= self.min_chunk_chars:
                logger.info("chunker flush | reason=final/packed | chars=%d | preview=%s", len(chunk), chunk[:80])
                yield chunk
            else:
                # keep it in buffer if you want; simplest is just emit
                logger.info("chunker flush | reason=small | chars=%d | preview=%s", len(chunk), chunk[:80])
                yield chunk


def buffered(iter_text: Iterable[str], *, language: Lang = "en", min_chars=25, max_chars=40) -> Iterator[str]:
    ch = StreamingSentenceChunker(language=language, min_chunk_chars=min_chars, max_chunk_chars=max_chars)
    for part in iter_text:
        yield from ch.feed(part)
    yield from ch.flush()
