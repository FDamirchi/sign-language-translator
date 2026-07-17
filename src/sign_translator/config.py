from dataclasses import dataclass, field


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


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    min_confidence: float = 0.51
    hold_seconds: float = 2.0
    release_seconds: float = 0.4
    neutral_labels: tuple[str, ...] = ("BACKGROUND",)


@dataclass(frozen=True, slots=True)
class FinalizationConfig:
    min_confidence: float = 0.80
    background_seconds: float = 5.0
    background_labels: tuple[str, ...] = ("BACKGROUND",)


@dataclass(frozen=True, slots=True)
class SpeechConfig:
    backend: str = "noop" # TODO: Change to pyttsx3 later
    lowercase_before_speaking: bool = True


@dataclass(frozen=True, slots=True)
class AppConfig:
    window_title: str = "Sign-Language Translator"
    predictor_backend: str = "mock"  # TODO: Change to the main model later

    quit_keys: tuple[int, ...] = (ord("q"), 27)

    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    finalization: FinalizationConfig = field(default_factory=FinalizationConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
