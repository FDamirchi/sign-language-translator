from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from sign_translator.config import (
    ModelInputConfig,
)
from sign_translator.inference.torch_predictor import (
    KerasPredictor,
    ModelLoadError,
)


class FakeKerasModel:
    def __init__(self) -> None:
        self.received_batch: NDArray[np.float32] | None = None

        self.training_value: bool | None = None

    def __call__(
        self,
        inputs: NDArray[np.float32],
        *,
        training: bool,
    ) -> NDArray[np.float32]:
        self.received_batch = inputs
        self.training_value = training

        return np.array(
            [[0.10, 0.80, 0.10]],
            dtype=np.float32,
        )


def test_predictor_preprocesses_and_decodes() -> None:
    fake_model = FakeKerasModel()

    predictor = KerasPredictor(
        model_path="unused.keras",
        labels=("A", "B", "C"),
        input_config=ModelInputConfig(),
        output_mode="probabilities",
        model=fake_model,
    )

    image = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    image[:, :] = [10, 20, 30]

    prediction = predictor.predict(image)

    assert prediction.label == "B"
    assert prediction.confidence == (pytest.approx(0.80))

    assert fake_model.received_batch is not None

    assert fake_model.received_batch.shape == (1, 200, 200, 3)

    assert fake_model.received_batch.dtype == np.float32

    assert fake_model.training_value is False

    np.testing.assert_allclose(
        fake_model.received_batch[
            0,
            0,
            0,
        ],
        np.array(
            [30, 20, 10],
            dtype=np.float32,
        )
        / 255.0,
    )


def test_missing_model_file_is_reported(
    tmp_path: Path,
) -> None:
    missing_model = tmp_path / "missing.keras"

    with pytest.raises(
        ModelLoadError,
        match="not found",
    ):
        KerasPredictor(
            model_path=missing_model,
            labels=("A", "B"),
        )
