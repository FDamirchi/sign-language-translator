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

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def resolve_roi(
    frame: NDArray[np.uint8],
    config: RoiConfig,
) -> RoiBox:
    if frame.ndim < 2:
        raise ValueError("Frame must have at least two dimensions")

    frame_height, frame_width = frame.shape[:2]

    region_x1 = int(frame_width * config.x_ratio)
    region_y1 = int(frame_height * config.y_ratio)

    region_width = int(frame_width * config.width_ratio)
    region_height = int(frame_height * config.height_ratio)

    region_x1 = max(
        0,
        min(
            region_x1,
            frame_width - 1,
        ),
    )

    region_y1 = max(
        0,
        min(
            region_y1,
            frame_height - 1,
        ),
    )

    region_width = max(
        1,
        min(
            region_width,
            frame_width - region_x1,
        ),
    )

    region_height = max(
        1,
        min(
            region_height,
            frame_height - region_y1,
        ),
    )

    if config.square:
        side = min(
            region_width,
            region_height,
        )

        x1 = region_x1 + (region_width - side) // 2
        y1 = region_y1 + (region_height - side) // 2

        x2 = x1 + side
        y2 = y1 + side

    else:
        x1 = region_x1
        y1 = region_y1
        x2 = region_x1 + region_width
        y2 = region_y1 + region_height

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
