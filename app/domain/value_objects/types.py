from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LanguageCode(str, Enum):
    EN_US = "en_us"
    EN_CA = "en_ca"
    PT_BR = "pt_br"

    @classmethod
    def from_str(cls, lang: str | None) -> "LanguageCode":
        if not lang:
            return cls.EN_US
        lang = lang.lower()
        if lang in ("en", "en-us", "en_us"):
            return cls.EN_US
        if lang in ("en-ca", "en_ca"):
            return cls.EN_CA
        if lang in ("pt", "pt-br", "pt_br"):
            return cls.PT_BR
        return cls.EN_US

class LanguageISO(str, Enum):
    EN_US = "en"
    EN_CA = "en"
    PT_BR = "en"

    @classmethod
    def from_str(cls, lang: str | None) -> "LanguageISO":
        if not lang:
            return cls.EN_US
        lang = lang.lower()
        if lang in ("en", "en-us", "en_us"):
            return cls.EN_US
        if lang in ("en-ca", "en_ca"):
            return cls.EN_CA
        if lang in ("pt", "pt-br", "pt_br"):
            return cls.PT_BR
        return cls.EN_US


class AudioFormat(str, Enum):
    WAV = "wav"
    AAC = "aac"
    ADTS = "adts"
    M4A = "m4a"

    @classmethod
    def from_str(cls, fmt: str | None) -> "AudioFormat":
        if not fmt:
            return cls.WAV
        fmt = fmt.lower()
        if fmt in ("aac",):
            return cls.AAC
        if fmt == "adts":
            return cls.ADTS
        if fmt in ("m4a", "ma4"):  # tolerate common typo
            return cls.M4A
        return cls.WAV
