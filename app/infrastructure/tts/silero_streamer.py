from __future__ import annotations

import os
import numpy as np
from typing import Generator, Iterable, Union

from app.domain.entities.audio import AudioFrame
from app.domain.services.tts_port import ITTSStreamer
from app.domain.value_objects.types import LanguageCode
import logging


logger = logging.getLogger(__name__)

# Voices examples
# https://oobabooga.github.io/silero-samples/


class SileroTTSAdapter(ITTSStreamer):
    """
    Streams PCM audio using Silero TTS.
    Note: requires torch and a Silero TTS model; will raise at init if missing.
    """

    _LANG_TO_SPEAKER = {
        LanguageCode.EN_US: ("en", None),  # speaker chosen at runtime
        LanguageCode.PT_BR: ("en", None),  # fallback to en
    }

    def __init__(self, device: str | None = None, sample_rate: int = 48000):
        try:
            import torch  # type: ignore
        except ImportError as exc:
            raise RuntimeError("SileroTTSAdapter requires torch installed") from exc

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sample_rate = sample_rate
        try:
            self.speed_factor = float(os.getenv("TTS_SILERO_SPEED", "1.0"))
        except ValueError:
            self.speed_factor = 1.0

        try:
            import omegaconf  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("SileroTTSAdapter requires the 'omegaconf' package (pip install omegaconf)") from exc

        try:
            result = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="en",
                speaker="v3_en",
            )
            # Some releases return (model, example_text, languages, speakers); others only (model, example_text)
            if isinstance(result, (list, tuple)):
                self.model = result[0]
                self._example_text = result[1] if len(result) > 1 else ""
                self._languages = result[2] if len(result) > 2 else {}
                self._speakers = result[3] if len(result) > 3 else []
            else:
                self.model = result
                self._example_text = ""
                self._languages = {}
                self._speakers = []
        except Exception as exc:
            raise RuntimeError("Failed to load Silero TTS model (check torch.hub connectivity/cache)") from exc

        # Normalize speaker list
        try:
            self._speakers = list(self.model.speakers)
        except Exception:
            if not hasattr(self, "_speakers") or not self._speakers:
                self._speakers = ["en_0"]
        # Allow overriding speaker via env (e.g., TTS_SPEAKER=en_10 for a deeper male voice)
        env_speaker = os.getenv("TTS_SPEAKER")
        if env_speaker and env_speaker in self._speakers:
            self.default_speaker = env_speaker
        else:
            self.default_speaker = self._speakers[0]

        self.model.to(self.device)

    def _speak(self, text: str, speaker: str) -> bytes:
        # Basic sanitization for silero; trim overly long inputs and strip control chars/newlines
        text = (text or "").replace("\n", " ").strip()
        if len(text) > 400:
            text = text[:400]

        try:
            audio = self.model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=self.sample_rate,
            )
        except Exception as exc:
            # Fail fast so the stream stops and returns 500
            raise RuntimeError(f"Silero TTS failed: {exc}") from exc

        arr = np.array(audio, dtype=np.float32)

        # Simple time-stretch: resample waveform to speed up (>1) or slow down (<1)
        if self.speed_factor and self.speed_factor != 1.0:
            new_len = max(1, int(len(arr) / self.speed_factor))
            x_old = np.linspace(0, 1, num=len(arr), endpoint=False)
            x_new = np.linspace(0, 1, num=new_len, endpoint=False)
            arr = np.interp(x_new, x_old, arr)

        pcm = (arr * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        return pcm

    def stream_pcm(self, text_or_iter: Union[str, Iterable[str]], language: LanguageCode, quality: str = "low") -> Generator[AudioFrame, None, None]:
        _lang_key, speaker = self._LANG_TO_SPEAKER.get(language, ("en", None))
        chosen_speaker = speaker or self.default_speaker  # fallback to the first available / env override

        if isinstance(text_or_iter, str):
            texts = [text_or_iter]
        else:
            texts = (t for t in text_or_iter if t)

        for text in texts:
            logger.info("silero.speak start | speaker=%s | chars=%d | text_preview=%s", chosen_speaker, len(text or ""), (text or "")[:80])
            pcm = self._speak(text, speaker=chosen_speaker)
            logger.info("silero.speak done | speaker=%s | bytes=%d", chosen_speaker, len(pcm))
            yield AudioFrame(
                sample_rate=self.sample_rate,
                channels=1,
                sample_width=2,
                data=pcm,
            )
