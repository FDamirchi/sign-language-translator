from dataclasses import dataclass, field
from pathlib import Path

from sign_translator.labels import MODEL_OUTPUT_LABELS, NOTHING_LABEL


@dataclass(frozen=True, slots=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    mirror: bool = True


@dataclass(frozen=True, slots=True)
class RoiConfig:
    x_ratio: float = 0.55
    y_ratio: float = 0.12
    width_ratio: float = 0.38
    height_ratio: float = 0.72
    square: bool = True


@dataclass(frozen=True, slots=True)
class ModelInputConfig:
    width: int = 200
    height: int = 200
    channels: int = 3
    color_space: str = "RGB"
    normalize_to_unit_interval: bool = True


@dataclass(frozen=True, slots=True)
class KerasPredictorConfig:
    model_path: Path = Path("models/asl_model.keras")

    output_mode: str = "auto"

    labels: tuple[str, ...] = MODEL_OUTPUT_LABELS


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    min_confidence: float = 0.51
    hold_seconds: float = 2.0
    release_seconds: float = 0.4

    neutral_labels: tuple[str, ...] = (NOTHING_LABEL,)


@dataclass(frozen=True, slots=True)
class FinalizationConfig:
    min_confidence: float = 0.80
    inactivity_seconds: float = 5.0

    neutral_labels: tuple[str, ...] = (NOTHING_LABEL,)


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    backend: str = "pyttsx3"
    lowercase_before_speaking: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    window_title: str = "Sign-Language Translator"
    predictor_backend: str = "mock"  # TODO: Change to the main model later

    quit_keys: tuple[int, ...] = (ord("q"), 27)

    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    model_input: ModelInputConfig = field(default_factory=ModelInputConfig)
    keras: KerasPredictorConfig = field(default_factory=KerasPredictorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    finalization: FinalizationConfig = field(default_factory=FinalizationConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
