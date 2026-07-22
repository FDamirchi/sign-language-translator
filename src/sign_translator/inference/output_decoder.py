from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from sign_translator.contracts import Prediction
from sign_translator.labels import normalize_label


class ModelOutputError(ValueError):
    """
    Raised when model output cannot be converted to a valid Prediction.
    """


VALID_OUTPUT_MODES = frozenset(
    {
        "auto",
        "probabilities",
        "logits",
    }
)


def decode_model_output(
    raw_output: object,
    labels: Sequence[str],
    *,
    output_mode: str = "auto",
) -> Prediction:
    canonical_labels = _normalize_labels(labels)

    scores = _extract_scores(raw_output)

    if scores.size != len(canonical_labels):
        raise ModelOutputError(
            "Model output class count does not match label count: "
            f"{scores.size} scores for "
            f"{len(canonical_labels)} labels!"
        )

    if not np.all(np.isfinite(scores)):
        raise ModelOutputError("Model output contains NaN or infinite values!")

    probabilities = _to_probabilities(scores, output_mode=output_mode)

    predicted_index = int(np.argmax(probabilities))

    predicted_label = canonical_labels[predicted_index]

    confidence = float(probabilities[predicted_index])

    return Prediction(
        label=predicted_label,
        confidence=confidence,
    )


def _normalize_labels(labels: Sequence[str]) -> tuple[str, ...]:
    canonical = tuple(normalize_label(label) for label in labels)

    if not canonical:
        raise ModelOutputError("At least one model label is required!")

    if len(set(canonical)) != len(canonical):
        raise ModelOutputError("Model labels must be unique!")

    return canonical


def _extract_scores(raw_output: object) -> NDArray[np.float64]:
    if isinstance(raw_output, (dict, list, tuple)):
        raise ModelOutputError(
            "Expected one output tensor, but model returned multiple outputs!"
        )

    try:
        array = np.asarray(raw_output, dtype=np.float64)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ModelOutputError(
            "Could not convert model output to a numeric array"
        ) from exc

    if array.ndim == 1:
        scores = array

    elif array.ndim == 2 and array.shape[0] == 1:
        scores = array[0]

    else:
        raise ModelOutputError(
            "Expected model output shape (classes,) or (1, classes), "
            f"got {array.shape}"
        )

    if scores.size == 0:
        raise ModelOutputError("Model output cannot be empty!")

    return scores


def _to_probabilities(
    scores: NDArray[np.float64],
    *,
    output_mode: str,
) -> NDArray[np.float64]:
    mode = output_mode.strip().lower()

    if mode not in VALID_OUTPUT_MODES:
        raise ModelOutputError(f"Unsupported output mode: " f"{output_mode!r}")

    if mode == "probabilities":
        return _normalize_probabilities(
            scores,
            require_probability_range=True,
        )

    if mode == "logits":
        return _softmax(scores)

    if _looks_like_probabilities(scores):
        return _normalize_probabilities(
            scores,
            require_probability_range=True,
        )

    return _softmax(scores)


def _looks_like_probabilities(scores: NDArray[np.float64]) -> bool:
    within_range = bool(np.all(scores >= -1e-7) and np.all(scores <= 1.0 + 1e-7))

    sums_to_one = bool(
        np.isclose(
            scores.sum(),
            1.0,
            atol=1e-3,
        )
    )

    return within_range and sums_to_one


def _normalize_probabilities(
    scores: NDArray[np.float64],
    *,
    require_probability_range: bool,
) -> NDArray[np.float64]:
    if require_probability_range and not _looks_like_probabilities(scores):
        raise ModelOutputError(
            "Output was declared as probabilities but values are outside [0, 1] or do not sum to 1!"
        )

    clipped = np.clip(scores, 0.0, 1.0)

    total = float(clipped.sum())

    if total <= 0.0:
        raise ModelOutputError("Probability sum must be greater than 0!")

    return clipped / total


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - np.max(logits)

    exponentials = np.exp(shifted)
    denominator = float(exponentials.sum())

    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ModelOutputError("Could not calculate softmax for model output!")

    return exponentials / denominator
