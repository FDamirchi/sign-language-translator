from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sign_translator.contracts import Prediction


class Predictor(Protocol):
    def predict(
        self,
        image: NDArray[np.uint8],
    ) -> Prediction:
        """
        Return one canonical label and a normalized
        confidence score.
        """
        ...
