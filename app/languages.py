from __future__ import annotations
from dataclasses import dataclass
from dataclasses import asdict
from typing import Dict

@dataclass(frozen=True)
class LangInfo:
    code: str           # canonical code like "en_us"
    name: str           # human label
    aliases: tuple[str, ...]
    voice_file: str     # Piper ONNX model
    length_scale: float


# Canonical registry
_LANGS: dict[str, LangInfo] = {
    "en_us": LangInfo(
        code="en_us",
        name="English (US)",
        aliases=("en", "en_us", "en-us", "english", "english_us", "en_us_utf8"),
        voice_file="en_US-ryan-medium.onnx",
        length_scale=1.0,   # 1.0 = normal, >1 = slower
         
    ),
    "pt_br": LangInfo(
        code="pt_br",
        name="Portuguese (Brazil)",
        aliases=("pt", "pt_br", "pt-br", "portugues", "português", "brazilian_portuguese"),
        voice_file="pt_BR-cadu-medium.onnx",  # change to a voice you actually have
        length_scale=0.8,   # 1.0 = normal, >1 = slower
    ),
}

def get_voice(acronym: str | None) -> str:
    code = (acronym or "").strip().lower().replace("-", "_")
    info = _LANGS.get(code)
    return info.voice_file if info else _LANGS["en_us"].voice_file

def get_scale(acronym: str | None) -> str:
    code = (acronym or "").strip().lower().replace("-", "_")
    info = _LANGS.get(code)
    return info.length_scale if info else _LANGS["en_us"].length_scale

def get_name(acronym: str | None) -> str:
    code = (acronym or "").strip().lower().replace("-", "_")
    info = _LANGS.get(code)
    return info.name if info else _LANGS["en_us"].name

def get_languages_json() -> Dict[str, dict]:
    """Canonical code -> plain dicts (safe to serialize)."""
    return {code: asdict(info) for code, info in _LANGS.items()}