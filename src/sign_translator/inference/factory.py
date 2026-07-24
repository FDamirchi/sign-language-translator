from sign_translator.config import KerasPredictorConfig, ModelInputConfig
from sign_translator.inference.base import Predictor
from sign_translator.inference.torch_predictor import KerasPredictor
from sign_translator.inference.mock import TimedMockPredictor


def create_predictor(
    backend: str,
    *,
    model_input: ModelInputConfig | None = None,
    keras_config: KerasPredictorConfig | None = None,
) -> Predictor:
    normalized_backend = backend.strip().lower()

    if normalized_backend == "mock":
        return TimedMockPredictor()

    if normalized_backend == "keras":
        resolved_input = model_input or ModelInputConfig()

        resolved_keras = keras_config or KerasPredictorConfig()

        return KerasPredictor(
            model_path=(resolved_keras.model_path),
            labels=(resolved_keras.labels),
            input_config=(resolved_input),
            output_mode=(resolved_keras.output_mode),
        )

    raise ValueError("Unsupported predictor backend: " f"{backend!r}")
