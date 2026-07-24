import torch

from sign_translator.inference.torch_model import (
    SignLanguageCNN,
)


def test_model_returns_29_logits() -> None:
    model = SignLanguageCNN(num_classes=29)

    model.eval()

    inputs = torch.zeros(
        (2, 3, 200, 200),
        dtype=torch.float32,
    )

    with torch.inference_mode():
        output = model(inputs)

    assert output.shape == (
        2,
        29,
    )
