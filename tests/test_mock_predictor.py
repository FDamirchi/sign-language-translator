import numpy as np

from sign_translator.inference.mock import (
    TimedMockPredictor,
)


def test_mock_predictor_returns_prediction() -> None:
    predictor = TimedMockPredictor(
        labels=("TEST",),
        seconds_per_label=1.0,
    )

    image = np.zeros(
        (128, 128, 3),
        dtype=np.uint8,
    )

    prediction = predictor.predict(image)

    assert prediction.label == "TEST"
    assert prediction.confidence == 1.0
