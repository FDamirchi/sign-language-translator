from collections.abc import Sequence
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
    *,
    mean: Sequence[float],
    std: Sequence[float],
) -> PreprocessedImage:
    input_config = config or ModelInputConfig()

    if image.ndim != 3 or image.shape[2] != input_config.channels:
        raise ImagePreprocessingError("Expected a three-channel HxWx3 image!")

    if image.size == 0:
        raise ImagePreprocessingError("Cannot preprocess an empty image!")

    if input_config.channels != 3:
        raise ImagePreprocessingError(
            "Only three-channel model input is currently supported!"
        )

    if input_config.color_space.strip().upper() != "RGB":
        raise ImagePreprocessingError("Only RGB model input is currently supported!")

    mean_array, std_array = _validate_normalization_stats(
        mean=mean,
        std=std,
        channels=input_config.channels,
    )

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

    normalized = (converted - mean_array.reshape(1, 1, 3)) / std_array.reshape(1, 1, 3)

    channels_first = np.transpose(normalized, (2, 0, 1))

    batch = np.expand_dims(channels_first, axis=0)

    batch = np.ascontiguousarray(batch, dtype=np.float32)

    return PreprocessedImage(batch=batch)


def _validate_normalization_stats(
    *,
    mean: Sequence[float],
    std: Sequence[float],
    channels: int,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.float32],
]:
    mean_array = np.asarray(mean, dtype=np.float32)

    std_array = np.asarray(std, dtype=np.float32)

    if mean_array.shape != (channels,):
        raise ImagePreprocessingError("Mean must contain one value per image channel!")

    if std_array.shape != (channels,):
        raise ImagePreprocessingError("Std must contain one value per image channel!")

    if not np.all(np.isfinite(mean_array)):
        raise ImagePreprocessingError("Mean contains invalid values!")

    if not np.all(np.isfinite(std_array)) or np.any(std_array <= 0):
        raise ImagePreprocessingError(
            "Std values must be finite and greater than zero!"
        )

    return mean_array, std_array


def _select_interpolation(
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> int:
    is_downscaling = target_width < source_width or target_height < source_height

    return cv2.INTER_AREA if is_downscaling else cv2.INTER_LINEAR
