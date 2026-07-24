from sign_translator.config import ModelInputConfig, TorchPredictorConfig
from sign_translator.inference.base import Predictor
from sign_translator.inference.mock import TimedMockPredictor


def create_predictor(
    backend: str,
    *,
    model_input: ModelInputConfig | None = None,
    torch_config: TorchPredictorConfig | None = None,
) -> Predictor:
    normalized_backend = backend.strip().lower()

    if normalized_backend == "mock":
        return TimedMockPredictor()

    if normalized_backend == "torch":
        from sign_translator.inference.torch_predictor import TorchPredictor

        resolved_input = model_input or ModelInputConfig()

        resolved_torch = torch_config or TorchPredictorConfig()

        return TorchPredictor(
            model_path=(resolved_torch.model_path),
            labels=(resolved_torch.labels),
            mean=(resolved_torch.mean),
            std=(resolved_torch.std),
            input_config=(resolved_input),
            device=(resolved_torch.device),
        )

    raise ValueError("Unsupported predictor backend: " f"{backend!r}")
