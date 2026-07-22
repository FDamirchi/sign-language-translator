import numpy as np
import pytest

from sign_translator.config import (
    ModelInputConfig,
)
from sign_translator.inference.preprocessing import (
    ImagePreprocessingError,
    preprocess_bgr_image,
)


def test_preprocesses_to_200_square_rgb_float_batch() -> None:
    bgr = np.zeros(
        (100, 300, 3),
        dtype=np.uint8,
    )

    bgr[:, :] = [10, 20, 30]

    result = preprocess_bgr_image(
        bgr,
        ModelInputConfig(),
    )

    assert result.batch.shape == (
        1,
        200,
        200,
        3,
    )

    assert result.batch.dtype == np.float32

    assert result.batch.min() >= 0.0
    assert result.batch.max() <= 1.0

    rgb_pixel = result.batch[
        0,
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


def test_rejects_non_three_channel_image() -> None:
    grayscale = np.zeros(
        (200, 200),
        dtype=np.uint8,
    )

    with pytest.raises(
        ImagePreprocessingError,
        match="three-channel",
    ):
        preprocess_bgr_image(grayscale)
