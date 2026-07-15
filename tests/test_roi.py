import numpy as np

from sign_translator.config import RoiConfig
from sign_translator.vision.roi import (
    crop_roi,
    resolve_roi,
)


def test_resolve_and_crop_roi() -> None:
    frame = np.zeros(
        (100, 200, 3),
        dtype=np.uint8,
    )

    config = RoiConfig(
        x_ratio=0.5,
        y_ratio=0.1,
        width_ratio=0.4,
        height_ratio=0.8,
    )

    box = resolve_roi(frame, config)
    roi = crop_roi(frame, box)

    assert box.x1 == 100
    assert box.y1 == 10
    assert box.x2 == 180
    assert box.y2 == 90

    assert roi.shape == (80, 80, 3)
