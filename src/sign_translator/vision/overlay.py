import cv2
import numpy as np
from numpy.typing import NDArray

from sign_translator.contracts import Prediction
from sign_translator.decoding.sequence_finalizer import FinalizationUpdate
from sign_translator.decoding.temporal_decoder import DecoderUpdate
from sign_translator.labels import NOTHING_LABEL
from sign_translator.vision.roi import RoiBox


def draw_overlay(
    frame: NDArray[np.uint8],
    box: RoiBox,
    prediction: Prediction,
    decoder_update: DecoderUpdate,
    finalization_update: FinalizationUpdate,
    current_text: str,
    last_finalized_text: str,
) -> None:
    cv2.rectangle(
        frame,
        (box.x1, box.y1),
        (box.x2, box.y2),
        (0, 255, 0),
        2,
    )

    text_x = box.x1
    text_y = max(
        30,
        box.y1 - 62,
    )

    _draw_text(
        frame,
        ("Prediction: " f"{prediction.label.upper()}"),
        text_x,
        text_y,
        scale=0.65,
    )

    _draw_text(
        frame,
        ("Confidence: " f"{prediction.confidence:.1%}"),
        text_x,
        text_y + 26,
        scale=0.55,
    )

    _draw_text(
        frame,
        _build_decoder_status(decoder_update),
        text_x,
        text_y + 52,
        scale=0.55,
        color=(0, 255, 255),
    )

    _draw_progress_bar(
        frame,
        x1=box.x1,
        y1=box.y2 + 12,
        x2=box.x2,
        progress=decoder_update.progress,
    )

    if finalization_update.neutral_active:
        _draw_text(
            frame,
            ("Finalizing text: " f"{int(finalization_update.progress * 100)}%"),
            box.x1,
            box.y2 + 54,
            scale=0.55,
            color=(255, 255, 0),
        )

        _draw_progress_bar(
            frame,
            x1=box.x1,
            y1=box.y2 + 64,
            x2=box.x2,
            progress=(finalization_update.progress),
        )

    displayed_current = current_text or "-"
    displayed_last = last_finalized_text or "-"

    _draw_text(
        frame,
        ("Current text: " f"{displayed_current}"),
        24,
        frame.shape[0] - 56,
        scale=0.75,
    )

    _draw_text(
        frame,
        ("Last text: " f"{displayed_last}"),
        24,
        frame.shape[0] - 26,
        scale=0.65,
        color=(200, 200, 200),
    )

    _draw_text(
        frame,
        "Press Q or ESC to quit",
        24,
        32,
        scale=0.55,
    )


def _build_decoder_status(
    update: DecoderUpdate,
) -> str:
    if update.accepted_label is not None:
        return "Accepted: " f"{update.accepted_label}"

    if update.is_locked:
        return "Locked - show " f"{NOTHING_LABEL}"

    if update.candidate_label is not None:
        percentage = int(update.progress * 100)

        return "Hold steady: " f"{percentage}%"

    if update.release_completed:
        return "Released"

    return "Waiting for a stable sign"


def _draw_progress_bar(
    frame: NDArray[np.uint8],
    *,
    x1: int,
    y1: int,
    x2: int,
    progress: float,
) -> None:
    y2 = y1 + 14

    if y2 >= frame.shape[0]:
        return

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (80, 80, 80),
        1,
    )

    width = x2 - x1
    progress_width = int(width * progress)

    if progress_width <= 0:
        return

    cv2.rectangle(
        frame,
        (x1, y1),
        (
            x1 + progress_width,
            y2,
        ),
        (0, 255, 255),
        -1,
    )


def _draw_text(
    frame: NDArray[np.uint8],
    text: str,
    x: int,
    y: int,
    *,
    scale: float,
    color: tuple[int, int, int] = (
        255,
        255,
        255,
    ),
) -> None:
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )
