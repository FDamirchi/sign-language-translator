import cv2
import numpy as np
from numpy.typing import NDArray

from sign_translator.contracts import Prediction
from sign_translator.vision.roi import RoiBox


def draw_overlay(
    frame: NDArray[np.uint8],
    box: RoiBox,
    prediction: Prediction,
) -> None:
    cv2.rectangle(
        frame,
        (box.x1, box.y1),
        (box.x2, box.y2),
        (0, 255, 0),
        2,
    )

    prediction_text = f"Prediction: {prediction.label}"
    confidence_text = f"Confidence: {prediction.confidence:.1%}"

    text_x = box.x1
    text_y = max(30, box.y1 - 36)

    cv2.putText(
        frame,
        prediction_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        confidence_text,
        (text_x, text_y + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
