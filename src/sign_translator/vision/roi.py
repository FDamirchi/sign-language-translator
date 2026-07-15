from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sign_translator.config import RoiConfig


@dataclass(frozen=True, slots=True)
class RoiBox:
    x1: int
    y1: int
    x2: int
    y2: int


def resolve_roi(
    frame: NDArray[np.uint8],
    config: RoiConfig,
) -> RoiBox:
    if frame.ndim < 2:
        raise ValueError("Frame must have at least two dimensions")

    frame_height, frame_width = frame.shape[:2]

    x1 = int(frame_width * config.x_ratio)
    y1 = int(frame_height * config.y_ratio)

    x2 = int(frame_width * (config.x_ratio + config.width_ratio))
    y2 = int(frame_height * (config.y_ratio + config.height_ratio))

    x1 = max(0, min(x1, frame_width - 1))
    y1 = max(0, min(y1, frame_height - 1))

    x2 = max(
        x1 + 1,
        min(x2, frame_width),
    )
    y2 = max(
        y1 + 1,
        min(y2, frame_height),
    )

    return RoiBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
    )


def crop_roi(
    frame: NDArray[np.uint8],
    box: RoiBox,
) -> NDArray[np.uint8]:
    return frame[
        box.y1 : box.y2,
        box.x1 : box.x2,
    ]
