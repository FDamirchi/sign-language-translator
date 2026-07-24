import numpy as np
import pytest

from sign_translator.config import (
    ModelInputConfig,
)
from sign_translator.inference.preprocessing import (
    ImagePreprocessingError,
    preprocess_bgr_image,
)


def test_preprocesses_to_nchw_rgb_batch() -> None:
    bgr = np.zeros(
        (100, 300, 3),
        dtype=np.uint8,
    )

    bgr[:, :] = [10, 20, 30]

    result = preprocess_bgr_image(
        bgr,
        ModelInputConfig(),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )

    assert result.batch.shape == (
        1,
        3,
        200,
        200,
    )

    assert result.batch.dtype == np.float32

    rgb_pixel = result.batch[
        0,
        :,
        0,
        0,
    ]

    np.testing.assert_allclose(
        rgb_pixel,
        np.array(
            [30, 20, 10],
            dtype=np.float32,
        )
        / 255.0,
    )


def test_applies_channel_normalization() -> None:
    image = np.full(
        (200, 200, 3),
        255,
        dtype=np.uint8,
    )

    mean = (
        0.5,
        0.25,
        0.75,
    )

    std = (
        0.5,
        0.25,
        0.25,
    )

    result = preprocess_bgr_image(
        image,
        mean=mean,
        std=std,
    )

    expected = np.array(
        [
            (1.0 - 0.5) / 0.5,
            (1.0 - 0.25) / 0.25,
            (1.0 - 0.75) / 0.25,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        result.batch[0, :, 0, 0],
        expected,
    )


def test_rejects_non_three_channel_image() -> None:
    grayscale = np.zeros(
        (200, 200),
        dtype=np.uint8,
    )

    with pytest.raises(
        ImagePreprocessingError,
        match="three-channel",
    ):
        preprocess_bgr_image(
            grayscale,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )


def test_rejects_invalid_std() -> None:
    image = np.zeros(
        (200, 200, 3),
        dtype=np.uint8,
    )

    with pytest.raises(
        ImagePreprocessingError,
        match="greater than zero",
    ):
        preprocess_bgr_image(
            image,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 0.0, 1.0),
        )
