from dataclasses import dataclass, field
from time import monotonic

import numpy as np
from numpy.typing import NDArray

from sign_translator.contracts import Prediction


@dataclass(slots=True)
class TimedMockPredictor:
    labels: tuple[str, ...] = (
        "A",
        "B",
        "C",
        "BACKGROUND",
        "BACKGROUND",
    )
    seconds_per_label: float = 3.0

    _started_at: float | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("Mock predictor labels cannot be empty")

        if self.seconds_per_label <= 0:
            raise ValueError("seconds_per_label must be greater than zero")

    def predict(
        self,
        image: NDArray[np.uint8],
    ) -> Prediction:
        if image.size == 0:
            raise ValueError("Predictor received an empty image")

        current_time = monotonic()

        if self._started_at is None:
            self._started_at = current_time

        elapsed = current_time - self._started_at

        index = int(elapsed / self.seconds_per_label) % len(self.labels)

        return Prediction(
            label=self.labels[index],
            confidence=1.0,
        )
