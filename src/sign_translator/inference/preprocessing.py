from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from sign_translator.config import ModelInputConfig


class ImagePreprocessingError(ValueError):
    """
    Raised when a camera ROI cannot be converted to model input.
    """


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    batch: NDArray[np.float32]

    @property
    def image(self) -> NDArray[np.float32]:
        return self.batch[0]


def preprocess_bgr_image(
    image: NDArray[np.uint8],
    config: ModelInputConfig | None = None,
) -> PreprocessedImage:
    input_config = config or ModelInputConfig()

    if image.ndim != 3 or image.shape[2] != input_config.channels:
        raise ImagePreprocessingError("Expected a three-channel HxWx3 image!")

    if image.size == 0:
        raise ImagePreprocessingError("Cannot preprocess an empty image!")

    if input_config.channels != 3:
        raise ImagePreprocessingError(
            "Only three-channel RGB model input is supported!"
        )

    if input_config.color_space.strip().upper() != "RGB":
        raise ImagePreprocessingError("Only RGB model input is supported!")

    resized = cv2.resize(
        image,
        (
            input_config.width,
            input_config.height,
        ),
        interpolation=_select_interpolation(
            source_width=image.shape[1],
            source_height=image.shape[0],
            target_width=input_config.width,
            target_height=input_config.height,
        ),
    )

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    converted = rgb.astype(np.float32)

    if input_config.normalize_to_unit_interval:
        converted /= 255.0

    batch = np.expand_dims(converted, axis=0)

    return PreprocessedImage(batch=batch)


def _select_interpolation(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> int:
    is_downscaling = target_width < source_width or target_height < source_height

    return cv2.INTER_AREA if is_downscaling else cv2.INTER_LINEAR
