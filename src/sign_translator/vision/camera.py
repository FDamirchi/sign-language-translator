from types import TracebackType
from typing import Self

import cv2
import numpy as np
from numpy.typing import NDArray

from sign_translator.config import CameraConfig


class CameraError(RuntimeError):
    """Raised when the camera cannot be opened or read."""


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def open(self) -> None:
        if self._capture is not None:
            return

        capture = cv2.VideoCapture(self._config.index)

        capture.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self._config.width,
        )
        capture.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self._config.height,
        )

        if not capture.isOpened():
            capture.release()

            raise CameraError(
                f"Could not open camera index "
                f"{self._config.index}. "
                "Check camera permissions or try another index."
            )

        self._capture = capture

    def read(self) -> NDArray[np.uint8]:
        if self._capture is None:
            raise CameraError("Camera is not open")

        success, frame = self._capture.read()

        if not success or frame is None: # type: ignore
            raise CameraError("Could not read a frame from the camera")

        return frame # type: ignore

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
