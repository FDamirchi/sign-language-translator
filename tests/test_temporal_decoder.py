from sign_translator.contracts import (
    Prediction,
)
from sign_translator.decoding.temporal_decoder import (
    TemporalDecoder,
)


def make_prediction(
    label: str,
    confidence: float = 1.0,
) -> Prediction:
    return Prediction(
        label=label,
        confidence=confidence,
    )


def test_accepts_label_after_hold_duration() -> None:
    decoder = TemporalDecoder(
        hold_seconds=2.0,
    )

    first = decoder.update(
        make_prediction("a"),
        now=0.0,
    )

    accepted = decoder.update(
        make_prediction("A"),
        now=2.0,
    )

    assert first.accepted_label is None
    assert accepted.accepted_label == "A"
    assert accepted.progress == 1.0


def test_does_not_repeat_held_label() -> None:
    decoder = TemporalDecoder(
        hold_seconds=2.0,
    )

    decoder.update(
        make_prediction("A"),
        now=0.0,
    )

    accepted = decoder.update(
        make_prediction("A"),
        now=2.0,
    )

    repeated = decoder.update(
        make_prediction("A"),
        now=4.0,
    )

    assert accepted.accepted_label == "A"
    assert repeated.accepted_label is None
    assert repeated.is_locked is True


def test_nothing_releases_previous_label() -> None:
    decoder = TemporalDecoder(
        hold_seconds=2.0,
        release_seconds=0.4,
    )

    decoder.update(
        make_prediction("A"),
        now=0.0,
    )

    decoder.update(
        make_prediction("A"),
        now=2.0,
    )

    decoder.update(
        make_prediction("nothing"),
        now=2.1,
    )

    released = decoder.update(
        make_prediction("NOTHING"),
        now=2.6,
    )

    decoder.update(
        make_prediction("A"),
        now=3.0,
    )

    accepted_again = decoder.update(
        make_prediction("A"),
        now=5.0,
    )

    assert released.release_completed is True
    assert accepted_again.accepted_label == "A"


def test_low_confidence_prediction_is_ignored() -> None:
    decoder = TemporalDecoder(
        min_confidence=0.80,
        hold_seconds=2.0,
    )

    update = decoder.update(
        make_prediction(
            "A",
            confidence=0.50,
        ),
        now=0.0,
    )

    assert update.candidate_label is None
    assert update.accepted_label is None
    assert update.progress == 0.0


def test_different_label_can_be_accepted() -> None:
    decoder = TemporalDecoder(
        hold_seconds=2.0,
    )

    decoder.update(
        make_prediction("A"),
        now=0.0,
    )

    first = decoder.update(
        make_prediction("A"),
        now=2.0,
    )

    decoder.update(
        make_prediction("B"),
        now=2.1,
    )

    second = decoder.update(
        make_prediction("B"),
        now=4.2,
    )

    assert first.accepted_label == "A"
    assert second.accepted_label == "B"
