# app/voices.py
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
from piper.voice import PiperVoice

# Pick the available config class across Piper versions
try:
    from piper.voice import InferenceConfig as PiperConfig
except ImportError:
    from piper.voice import SynthesisConfig as PiperConfig  # older/newer alt

DEFAULT_VOICES_DIR = Path(__file__).resolve().parents[1] / "voices"

def _resolve(base_dir: Optional[str | Path], maybe_path: str) -> str:
    p = Path(maybe_path)
    if p.is_absolute():
        return str(p)
    root = Path(base_dir) if base_dir else DEFAULT_VOICES_DIR
    return str((root / p).resolve())

def _require(model_path: str):
    cfg_path = model_path + ".json"
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Piper model not found: {model_path}")
    if not Path(cfg_path).is_file():
        raise FileNotFoundError(f"Piper config not found: {cfg_path} (must be next to the .onnx)")

def make_config(length_scale=1.08, noise_scale=0.4, noise_w=0.85) -> PiperConfig:
    return PiperConfig(length_scale=length_scale, noise_scale=noise_scale)

class VoiceRegistry:
    def __init__(self):
        self._voices: Dict[str, PiperVoice] = {}
        self._cfgs: Dict[str, PiperConfig] = {}

    def load_all(self, base_dir: Optional[str | Path] = None) -> None:
        from app.languages import get_voice  # defer import to avoid cycles
        en_model = get_voice("en_us") or "en_US-ryan-medium.onnx"
        pt_model = get_voice("pt_br") or "pt_BR-faber-medium.onnx"

        en_path = _resolve(base_dir, en_model)
        pt_path = _resolve(base_dir, pt_model)

        _require(en_path); _require(pt_path)

        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        os.environ.setdefault("ORT_DISABLE_CUDA", "1")  # silences CUDA provider warning

        self._voices["en_us"] = PiperVoice.load(en_path, use_cuda=False)
        self._voices["pt_br"] = PiperVoice.load(pt_path, use_cuda=False)

        self._cfgs["en_us"] = make_config()
        self._cfgs["pt_br"] = make_config()

        # Pre-warm (note: pass CONFIG OBJECT, not kwargs)
        for k, v in self._voices.items():
            list(v.synthesize("Hi.", self._cfgs[k]))

    def _resolve_lang(self, lang: str) -> str:
        lang = (lang or "en_us").lower()
        if lang in ("en", "en-us", "en_us"): return "en_us"
        if lang in ("pt", "pt-br", "pt_br"): return "pt_br"
        return "en_us"

    def get(self, lang: str) -> Tuple[PiperVoice, PiperConfig]:
        key = self._resolve_lang(lang)
        return self._voices[key], self._cfgs[key]

    def list_langs(self):
        return list(self._voices.keys())
