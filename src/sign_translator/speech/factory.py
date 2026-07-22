from sign_translator.speech.base import SpeechEngine
from sign_translator.speech.noop import NoOpSpeechEngine
from sign_translator.speech.pyttsx3_engine import (
    Pyttsx3SpeechEngine,
)


def create_speech_engine(backend: str) -> SpeechEngine:
    normalized_backend = backend.strip().lower()

    if normalized_backend == "noop":
        return NoOpSpeechEngine()

    if normalized_backend == "pyttsx3":
        return Pyttsx3SpeechEngine()

    raise ValueError(f"Unsupported speech backend: {backend!r}")
