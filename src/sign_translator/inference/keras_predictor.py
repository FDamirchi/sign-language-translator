from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from sign_translator.config import ModelInputConfig
from sign_translator.contracts import Prediction
from sign_translator.inference.output_decoder import (
    ModelOutputError,
    decode_model_output,
)
from sign_translator.inference.preprocessing import preprocess_bgr_image


class KerasModelProtocol(Protocol):
    def __call__(
        self,
        inputs: NDArray[np.float32],
        *,
        training: bool,
    ) -> object: ...


class ModelLoadError(RuntimeError):
    """
    Raised when a Keras model cannot be loaded.
    """


class KerasPredictor:
    def __init__(
        self,
        *,
        model_path: str | Path,
        labels: Sequence[str],
        input_config: ModelInputConfig | None = None,
        output_mode: str = "auto",
        model: KerasModelProtocol | None = None,
    ) -> None:
        self._model_path = Path(model_path)
        self._labels = tuple(labels)
        self._input_config = input_config or ModelInputConfig()
        self._output_mode = output_mode
        self._model = model if model is not None else self._load_model()

    def predict(
        self,
        image: NDArray[np.uint8],
    ) -> Prediction:
        preprocessed = preprocess_bgr_image(
            image,
            self._input_config,
        )

        try:
            raw_output = self._model(
                preprocessed.batch,
                training=False,
            )
        except Exception as exc:
            raise RuntimeError("Keras model inference failed") from exc

        try:
            return decode_model_output(
                raw_output,
                self._labels,
                output_mode=(self._output_mode),
            )
        except ModelOutputError:
            raise

    def _load_model(
        self,
    ) -> KerasModelProtocol:
        if not self._model_path.is_file():
            raise ModelLoadError(
                "Keras model file was not found: " f"{self._model_path}"
            )

        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ModelLoadError(
                "TensorFlow is not installed.\nInstall the model dependencies with: "
                'python -m pip install -e ".[model]"'
            ) from exc

        try:
            loaded_model = keras.models.load_model(self._model_path, compile=False) # type: ignore
        except Exception as exc:
            raise ModelLoadError(
                "Could not load Keras model: " f"{self._model_path}"
            ) from exc

        return cast(
            KerasModelProtocol,
            loaded_model,
        )
