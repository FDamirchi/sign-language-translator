import cv2
import numpy as np
from numpy.typing import NDArray

from sign_translator.contracts import Prediction
from sign_translator.vision.roi import RoiBox
from sign_translator.decoding.temporal_decoder import DecoderUpdate


def draw_overlay(
    frame: NDArray[np.uint8],
    box: RoiBox,
    prediction: Prediction,
    decoder_update: DecoderUpdate,
    translated_text: str,
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
    status_text = _build_status_text(decoder_update)

    text_x = box.x1
    text_y = max(30, box.y1 - 62)

    cv2.putText(
        frame,
        prediction_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        confidence_text,
        (text_x, text_y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        status_text,
        (text_x, text_y + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    _draw_progress_bar(
        frame,
        box,
        decoder_update.progress,
    )

    displayed_text = translated_text or "-"

    cv2.putText(
        frame,
        f"Text: {displayed_text}",
        (24, frame.shape[0] - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Press Q or ESC to quit",
        (24, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _build_status_text(
    update: DecoderUpdate,
) -> str:
    if update.accepted_label is not None:
        return f"Accepted: {update.accepted_label}"

    if update.is_locked:
        return "Locked - show BACKGROUND"

    if update.candidate_label is not None:
        percentage = int(update.progress * 100)
        return f"Hold steady: {percentage}%"

    if update.release_completed:
        return "Released"

    return "Waiting for stable sign"


def _draw_progress_bar(
    frame: NDArray[np.uint8],
    box: RoiBox,
    progress: float,
) -> None:
    bar_x1 = box.x1
    bar_y1 = box.y2 + 12
    bar_x2 = box.x2
    bar_y2 = bar_y1 + 14

    if bar_y2 >= frame.shape[0]:
        return

    cv2.rectangle(
        frame,
        (bar_x1, bar_y1),
        (bar_x2, bar_y2),
        (80, 80, 80),
        1,
    )

    available_width = bar_x2 - bar_x1
    progress_width = int(available_width * progress)

    if progress_width > 0:
        cv2.rectangle(
            frame,
            (bar_x1, bar_y1),
            (bar_x1 + progress_width, bar_y2),
            (0, 255, 255),
            -1,
        )
