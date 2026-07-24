from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sign_translator.config import ModelInputConfig
from sign_translator.contracts import Prediction
from sign_translator.inference.output_decoder import decode_model_output
from sign_translator.inference.preprocessing import preprocess_bgr_image
from sign_translator.inference.torch_model import SignLanguageCNN


class ModelLoadError(RuntimeError):
    """
    Raised when a PyTorch model cannot be loaded.
    """


class TorchPredictor:
    def __init__(
        self,
        *,
        model_path: str | Path,
        labels: Sequence[str],
        mean: Sequence[float],
        std: Sequence[float],
        input_config: ModelInputConfig | None = None,
        device: str = "auto",
        model: nn.Module | None = None,
    ) -> None:
        self._model_path = Path(model_path)

        self._labels = tuple(labels)

        if not self._labels:
            raise ValueError("At least one model label is required!")

        self._mean = tuple(float(value) for value in mean)

        self._std = tuple(float(value) for value in std)

        self._input_config = input_config or ModelInputConfig()

        self._device = self._resolve_device(device)

        self._model = model if model is not None else self._load_model()
        self._model.to(self._device)
        self._model.eval()

    def predict(
        self,
        image: NDArray[np.uint8],
    ) -> Prediction:
        preprocessed = preprocess_bgr_image(
            image,
            self._input_config,
            mean=self._mean,
            std=self._std,
        )

        batch = torch.from_numpy(preprocessed.batch).to(
            self._device,
            non_blocking=True,
        )

        try:
            with torch.inference_mode():
                logits = self._model(batch)
        except Exception as exc:
            raise RuntimeError("PyTorch model inference failed!") from exc

        if not isinstance(
            logits,
            torch.Tensor,
        ):
            raise RuntimeError("PyTorch model did not return a tensor!")

        raw_output = logits.detach().cpu().numpy()

        return decode_model_output(
            raw_output,
            self._labels,
            output_mode="logits",
        )

    def _load_model(
        self,
    ) -> nn.Module:
        if not self._model_path.is_file():
            raise ModelLoadError(
                "PyTorch model file was not found: " f"{self._model_path}"
            )

        model = SignLanguageCNN(num_classes=len(self._labels))

        try:
            state_dict = torch.load(
                self._model_path,
                map_location=self._device,
                weights_only=True,
            )
        except Exception as exc:
            raise ModelLoadError(
                "Could not read PyTorch model weights: " f"{self._model_path}"
            ) from exc

        try:
            model.load_state_dict(
                state_dict,
                strict=True,
            )
        except Exception as exc:
            raise ModelLoadError(
                "Model weights do not match SignLanguageCNN architecture!"
            ) from exc

        return model

    @staticmethod
    def _resolve_device(
        requested_device: str,
    ) -> torch.device:
        normalized = requested_device.strip().lower()

        if normalized == "auto":
            normalized = "cuda" if torch.cuda.is_available() else "cpu"

        if normalized.startswith("cuda") and not torch.cuda.is_available():
            raise ModelLoadError("CUDA was requested but is not available!")

        try:
            return torch.device(normalized)
        except Exception as exc:
            raise ModelLoadError(
                "Invalid PyTorch device: " f"{requested_device!r}"
            ) from exc
