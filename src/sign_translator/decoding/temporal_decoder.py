from dataclasses import dataclass
from time import monotonic

from sign_translator.contracts import Prediction
from sign_translator.labels import NOTHING_LABEL, normalize_label


@dataclass(frozen=True, slots=True)
class DecoderUpdate:
    candidate_label: str | None
    accepted_label: str | None
    progress: float
    is_locked: bool
    release_completed: bool


class TemporalDecoder:
    def __init__(
        self,
        *,
        min_confidence: float = 0.51,
        hold_seconds: float = 2.0,
        release_seconds: float = 0.4,
        neutral_labels: tuple[str, ...] = (NOTHING_LABEL,),
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0!")

        if hold_seconds <= 0:
            raise ValueError("hold_seconds must be greater than 0!")

        if release_seconds < 0:
            raise ValueError("release_seconds cannot be negative!")

        self._min_confidence = min_confidence
        self._hold_seconds = hold_seconds
        self._release_seconds = release_seconds

        self._neutral_labels = frozenset(
            normalize_label(label) for label in neutral_labels
        )

        self._candidate_label: str | None = None
        self._candidate_since: float | None = None

        self._latched_label: str | None = None
        self._release_since: float | None = None

        self._last_timestamp: float | None = None

    def update(
        self,
        prediction: Prediction,
        *,
        now: float | None = None,
    ) -> DecoderUpdate:
        timestamp = monotonic() if now is None else now

        self._validate_timestamp(timestamp)

        label = normalize_label(prediction.label)

        is_neutral = (
            prediction.confidence < self._min_confidence
            or label in self._neutral_labels
        )

        if is_neutral:
            return self._handle_neutral(timestamp)

        return self._handle_sign(
            label,
            timestamp,
        )

    def reset(self) -> None:
        self._candidate_label = None
        self._candidate_since = None
        self._latched_label = None
        self._release_since = None
        self._last_timestamp = None

    def _handle_neutral(
        self,
        timestamp: float,
    ) -> DecoderUpdate:
        self._candidate_label = None
        self._candidate_since = None

        if self._release_since is None:
            self._release_since = timestamp

        release_elapsed = timestamp - self._release_since
        release_completed = False

        if release_elapsed >= self._release_seconds and self._latched_label is not None:
            self._latched_label = None
            release_completed = True

        return DecoderUpdate(
            candidate_label=None,
            accepted_label=None,
            progress=0.0,
            is_locked=False,
            release_completed=release_completed,
        )

    def _handle_sign(
        self,
        label: str,
        timestamp: float,
    ) -> DecoderUpdate:
        self._release_since = None

        if label != self._candidate_label:
            self._candidate_label = label
            self._candidate_since = timestamp

        if self._candidate_since is None:
            self._candidate_since = timestamp

        elapsed = timestamp - self._candidate_since
        progress = min(elapsed / self._hold_seconds, 1.0)

        is_locked = label == self._latched_label
        accepted_label: str | None = None

        if progress >= 1.0 and not is_locked:
            accepted_label = label
            self._latched_label = label
            is_locked = True

        return DecoderUpdate(
            candidate_label=label,
            accepted_label=accepted_label,
            progress=progress,
            is_locked=is_locked,
            release_completed=False,
        )

    def _validate_timestamp(
        self,
        timestamp: float,
    ) -> None:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Timestamps must be monotonically increasing!")

        self._last_timestamp = timestamp
