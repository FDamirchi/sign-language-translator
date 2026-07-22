from sign_translator.contracts import Prediction
from sign_translator.decoding.sequence_finalizer import (
    SequenceFinalizer,
)


def make_prediction(
    label: str,
    confidence: float = 0.95,
) -> Prediction:
    return Prediction(
        label=label,
        confidence=confidence,
    )


def test_finalizes_after_background_duration() -> None:
    finalizer = SequenceFinalizer(
        background_seconds=5.0,
    )

    first = finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=True,
        now=10.0,
    )

    completed = finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=True,
        now=15.0,
    )

    assert first.finalized is False
    assert completed.finalized is True
    assert completed.progress == 1.0


def test_does_not_finalize_twice_during_same_background() -> None:
    finalizer = SequenceFinalizer(
        background_seconds=5.0,
    )

    finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=True,
        now=0.0,
    )

    first = finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=True,
        now=5.0,
    )

    repeated = finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=False,
        now=8.0,
    )

    assert first.finalized is True
    assert repeated.finalized is False


def test_non_background_resets_timer() -> None:
    finalizer = SequenceFinalizer(
        background_seconds=5.0,
    )

    finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=True,
        now=0.0,
    )

    finalizer.update(
        make_prediction("A"),
        has_content=True,
        now=3.0,
    )

    restarted = finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=True,
        now=4.0,
    )

    not_completed = finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=True,
        now=7.0,
    )

    assert restarted.progress == 0.0
    assert not_completed.finalized is False


def test_low_confidence_background_does_not_count() -> None:
    finalizer = SequenceFinalizer(
        min_confidence=0.80,
        background_seconds=5.0,
    )

    update = finalizer.update(
        make_prediction(
            "BACKGROUND",
            confidence=0.40,
        ),
        has_content=True,
        now=0.0,
    )

    assert update.background_active is False
    assert update.progress == 0.0
    assert update.finalized is False


def test_empty_buffer_is_not_finalized() -> None:
    finalizer = SequenceFinalizer(
        background_seconds=5.0,
    )

    finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=False,
        now=0.0,
    )

    update = finalizer.update(
        make_prediction("BACKGROUND"),
        has_content=False,
        now=10.0,
    )

    assert update.finalized is False
    assert update.progress == 0.0
