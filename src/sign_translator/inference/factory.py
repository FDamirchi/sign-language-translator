from sign_translator.inference.base import Predictor
from sign_translator.inference.mock import TimedMockPredictor


def create_predictor(backend: str) -> Predictor:
    normalized_backend = backend.strip().lower()

    if normalized_backend == "mock":
        return TimedMockPredictor()

    raise ValueError(f"Unsupported predictor backend: {backend!r}")
