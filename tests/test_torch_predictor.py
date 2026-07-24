from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from sign_translator.config import (
    ModelInputConfig,
)
from sign_translator.inference.torch_predictor import (
    ModelLoadError,
    TorchPredictor,
)


class FakeTorchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.received_shape: tuple[int, ...] | None = None

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        self.received_shape = tuple(inputs.shape)

        return torch.tensor(
            [[0.1, 4.0, 0.2]],
            dtype=torch.float32,
            device=inputs.device,
        )


def test_predictor_preprocesses_and_decodes() -> None:
    fake_model = FakeTorchModel()

    predictor = TorchPredictor(
        model_path="unused.pth",
        labels=("A", "B", "C"),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        input_config=ModelInputConfig(),
        device="cpu",
        model=fake_model,
    )

    image = np.zeros(
        (300, 300, 3),
        dtype=np.uint8,
    )

    prediction = predictor.predict(image)

    assert prediction.label == "B"

    assert 0.0 <= prediction.confidence <= 1.0

    assert fake_model.received_shape == (
        1,
        3,
        200,
        200,
    )


def test_missing_model_file_is_reported(
    tmp_path: Path,
) -> None:
    missing_model = tmp_path / "missing.pth"

    with pytest.raises(
        ModelLoadError,
        match="not found",
    ):
        TorchPredictor(
            model_path=missing_model,
            labels=("A", "B"),
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
            device="cpu",
        )
