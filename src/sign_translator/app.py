import cv2

from sign_translator.config import AppConfig
from sign_translator.inference.factory import create_predictor
from sign_translator.vision.camera import Camera, CameraError
from sign_translator.vision.overlay import draw_overlay
from sign_translator.vision.roi import crop_roi, resolve_roi


def run(config: AppConfig | None = None) -> None:
    app_config = config or AppConfig()

    predictor = create_predictor(app_config.predictor_backend) # type: ignore

    try:
        with Camera(app_config.camera) as camera:
            while True:
                frame = camera.read()

                if app_config.camera.mirror:
                    frame = cv2.flip(frame, 1)

                roi_box = resolve_roi(frame, app_config.roi) # type: ignore
                roi_image = crop_roi(frame, roi_box) # type: ignore

                prediction = predictor.predict(roi_image)

                draw_overlay(frame, roi_box, prediction) # type: ignore

                cv2.imshow(app_config.window_title, frame)

                key = cv2.waitKey(1) & 0xFF

                if key in app_config.quit_keys:
                    break

    except CameraError as exc:
        raise SystemExit(f"Camera error: {exc}") from exc

    finally:
        cv2.destroyAllWindows()
