from __future__ import annotations

from app.infrastructure.config.voices import VoiceRegistry

def build_registry() -> VoiceRegistry:
    reg = VoiceRegistry()
    reg.load_all()
    return reg
