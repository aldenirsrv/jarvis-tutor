import re
from typing import Iterable, Iterator
import logging

logger = logging.getLogger(__name__)

# Matches one "complete" sentence ending with ., !, or ?
_SENTENCE_MATCH = re.compile(r'^\s*([^.!?]*[.!?])\s*')

def _process_sentence(s: str) -> str:
    s = s.strip()
    if not s:
        return ""

    # Avoid "..." artifacts that can produce audible clicks; normalize to "."
    s = s.replace("...", ".")

    words = s.split()
    n = len(words)

    if n <= 5:
        s += "."
    elif 6 <= n <= 10 and not s.endswith("."):
        s += "."
    elif n >= 11 and "\n" not in s:
        mid = n // 2
        s = " ".join(words[:mid]) + ", " + " ".join(words[mid:])

    return s + " "  # keep natural spacing after each sentence

def add_natural_pauses(iter_text: Iterable[str]) -> Iterator[str]:
    """
    Consume an iterable/generator of text chunks and yield processed chunks
    with 'natural pauses' inserted. Keeps an internal buffer so we only
    emit when a full sentence is available.
    """
    buf = ""
    for chunk in iter_text:
        if not chunk:
            continue
        buf += chunk

        # Emit complete sentences from the buffer
        while True:
            m = _SENTENCE_MATCH.match(buf)
            if not m:
                break
            sentence = m.group(1)
            yield _process_sentence(sentence)
            buf = buf[m.end():]  # drop emitted part

    # Flush any remaining partial text at the end
    tail = buf.strip()
    if tail:
        yield _process_sentence(tail)
