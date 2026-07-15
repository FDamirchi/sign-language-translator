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
class AppConfig:
    window_title: str = "Sign-Language Translator"
    predictor_backend: str = "mock"  # TODO: Change to the main model later

    quit_keys: tuple[int, ...] = (ord("q"), 27)

    camera: CameraConfig = field(default_factory=CameraConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
