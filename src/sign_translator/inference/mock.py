from dataclasses import dataclass, field
from time import monotonic

import numpy as np
from numpy.typing import NDArray

from sign_translator.contracts import Prediction


@dataclass(slots=True)
class TimedMockPredator:
    labels: tuple[str, ...] = (
        "A",
        "B",
        "C",
        "BACKGROUND",
    )

    seconds_per_label: float = 2.0
    _started_at: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError("Mock predictor labels cannot be empty!")

        self._started_at = monotonic()

    def predict(
        self,
        image: NDArray[np.uint8],
    ) -> Prediction:
        if image.size == 0:
            raise ValueError("Predictor recieved an empty image!")

        elapsed = monotonic() - self._started_at
        index = int(elapsed / self.seconds_per_label) % len(self.labels)

        return Prediction(label=self.labels[index], confidence=1.0)
