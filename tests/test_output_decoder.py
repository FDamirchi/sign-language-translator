import numpy as np
import pytest

from sign_translator.inference.output_decoder import (
    ModelOutputError,
    decode_model_output,
)


def test_decodes_probability_output() -> None:
    output = np.array(
        [[0.10, 0.80, 0.10]],
        dtype=np.float32,
    )

    prediction = decode_model_output(
        output,
        ("A", "B", "C"),
        output_mode="probabilities",
    )

    assert prediction.label == "B"
    assert prediction.confidence == (pytest.approx(0.80))


def test_decodes_logits_with_softmax() -> None:
    logits = np.array(
        [0.0, 2.0, 1.0],
        dtype=np.float32,
    )

    prediction = decode_model_output(
        logits,
        ("A", "B", "C"),
        output_mode="logits",
    )

    expected = np.exp(2.0) / (np.exp(0.0) + np.exp(2.0) + np.exp(1.0))

    assert prediction.label == "B"
    assert prediction.confidence == (pytest.approx(expected))


def test_auto_mode_detects_probabilities() -> None:
    output = np.array(
        [0.05, 0.15, 0.80],
        dtype=np.float32,
    )

    prediction = decode_model_output(
        output,
        ("A", "B", "C"),
        output_mode="auto",
    )

    assert prediction.label == "C"
    assert prediction.confidence == (pytest.approx(0.80))


def test_rejects_wrong_class_count() -> None:
    output = np.array(
        [0.2, 0.8],
        dtype=np.float32,
    )

    with pytest.raises(
        ModelOutputError,
        match="class count",
    ):
        decode_model_output(
            output,
            ("A", "B", "C"),
        )


def test_rejects_nan_output() -> None:
    output = np.array(
        [0.1, np.nan, 0.9],
        dtype=np.float32,
    )

    with pytest.raises(
        ModelOutputError,
        match="NaN",
    ):
        decode_model_output(
            output,
            ("A", "B", "C"),
        )


def test_rejects_duplicate_labels() -> None:
    output = np.array(
        [0.2, 0.8],
        dtype=np.float32,
    )

    with pytest.raises(
        ModelOutputError,
        match="unique",
    ):
        decode_model_output(
            output,
            ("A", "A"),
        )


def test_rejects_multiple_model_outputs() -> None:
    output = [
        np.array(
            [[0.2, 0.8]],
            dtype=np.float32,
        ),
        np.array(
            [[0.5]],
            dtype=np.float32,
        ),
    ]

    with pytest.raises(
        ModelOutputError,
        match="multiple outputs",
    ):
        decode_model_output(
            output,
            ("A", "B"),
        )
