from dataclasses import dataclass
from time import monotonic

from sign_translator.contracts import Prediction
from sign_translator.labels import NOTHING_LABEL, normalize_label


@dataclass(frozen=True, slots=True)
class FinalizationUpdate:
    neutral_active: bool
    progress: float
    finalized: bool

    @property
    def background_active(self) -> bool:
        return self.neutral_active


class SequenceFinalizer:
    def __init__(
        self,
        *,
        min_confidence: float = 0.80,
        inactivity_seconds: float = 5.0,
        neutral_labels: tuple[str, ...] = (NOTHING_LABEL,),
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0!")

        if inactivity_seconds <= 0:
            raise ValueError("inactivity_seconds must be greater than 0!")

        if not neutral_labels:
            raise ValueError("At least one neutral label is required!")

        self._min_confidence = min_confidence
        self._inactivity_seconds = inactivity_seconds

        self._neutral_labels = frozenset(
            normalize_label(label) for label in neutral_labels
        )

        self._neutral_since: float | None = None
        self._finalized_during_current_neutral_period = False
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

        label = normalize_label(prediction.label)

        is_neutral = (
            prediction.confidence >= self._min_confidence
            and label in self._neutral_labels
        )

        if not is_neutral:
            self._reset_neutral_state()

            return FinalizationUpdate(
                neutral_active=False,
                progress=0.0,
                finalized=False,
            )

        if self._finalized_during_current_neutral_period:
            return FinalizationUpdate(
                neutral_active=True,
                progress=1.0,
                finalized=False,
            )

        if not has_content:
            self._neutral_since = None

            return FinalizationUpdate(
                neutral_active=True,
                progress=0.0,
                finalized=False,
            )

        if self._neutral_since is None:
            self._neutral_since = timestamp

        elapsed = timestamp - self._neutral_since

        progress = min(elapsed / self._inactivity_seconds, 1.0)

        finalized = progress >= 1.0

        if finalized:
            self._finalized_during_current_neutral_period = True

        return FinalizationUpdate(
            neutral_active=True,
            progress=progress,
            finalized=finalized,
        )

    def reset(self) -> None:
        self._neutral_since = None
        self._finalized_during_current_neutral_period = False
        self._last_timestamp = None

    def _reset_neutral_state(self) -> None:
        self._neutral_since = None
        self._finalized_during_current_neutral_period = False

    def _validate_timestamp(
        self,
        timestamp: float,
    ) -> None:
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("Timestamps must be monotonically increasing!")

        self._last_timestamp = timestamp
