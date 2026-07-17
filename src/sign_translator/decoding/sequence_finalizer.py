from dataclasses import dataclass
from time import monotonic

from sign_translator.contracts import Prediction


@dataclass(frozen=True, slots=True)
class FinalizationUpdate:
    background_active: bool
    progress: float
    finalized: bool


class SequenceFinalizer:
    def __init__(
        self,
        *,
        min_confidence: float = 0.51,
        background_seconds: float = 5.0,
        background_labels: tuple[str, ...] = ("BACKGROUND",),
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0!")

        if background_seconds <= 0:
            raise ValueError("background_seconds must be greater than 0!")

        if not background_labels:
            raise ValueError("At least one background_label is required!")

        self._min_confidence = min_confidence
        self._background_seconds = background_seconds
        self._background_labels = frozenset(
            label.strip().upper() for label in background_labels
        )

        self._background_since: float | None = None
        self._finalized_during_current_background = False
        self._last_timestamp: float | None = None

    def update(
        self,
        prediction: Prediction,
        *,
        has_content: bool,
        now: float | None = None,
    ) -> FinalizationUpdate:
        timestamp = monotonic() if now is None else now
        self._validate_timestamp(timestamp)

        label = prediction.label.strip().upper()

        is_background = (
            prediction.confidence >= self._min_confidence
            and label in self._background_labels
        )

        if not is_background:
            self._reset_background_state()

            return FinalizationUpdate(
                background_active=False,
                progress=0.0,
                finalized=False,
            )

        if self._finalized_during_current_background:
            return FinalizationUpdate(
                background_active=True,
                progress=1.0,
                finalized=False,
            )

        if not has_content:
            self._background_since = None

            return FinalizationUpdate(
                background_active=True,
                progress=0.0,
                finalized=False,
            )

        if self._background_since is None:
            self._background_since = timestamp

        elapsed = timestamp - self._background_since
        progress = min(elapsed / self._background_seconds, 1.0)

        finalized = progress >= 1.0

        if finalized:
            self._finalized_during_current_background = True

        return FinalizationUpdate(
            background_active=True,
            progress=progress,
            finalized=finalized,
        )

    def reset(self) -> None:
        self._background_since = None
        self._finalized_during_current_background = False
        self._last_timestamp = None

    def _reset_background_state(self) -> None:
        self._background_since = None
        self._finalized_during_current_background = False

    def _validate_timestamp(self, timestamp: float) -> None:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Timestamp cannot be monotonically increasing!")

        self._last_timestamp = timestamp
