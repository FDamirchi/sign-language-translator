from time import monotonic

import cv2

from sign_translator.config import AppConfig
from sign_translator.decoding.sequence_finalizer import SequenceFinalizer
from sign_translator.decoding.temporal_decoder import TemporalDecoder
from sign_translator.decoding.text_buffer import TextBuffer
from sign_translator.inference.factory import create_predictor
from sign_translator.speech.factory import create_speech_engine
from sign_translator.vision.camera import Camera, CameraError
from sign_translator.vision.overlay import draw_overlay
from sign_translator.vision.roi import crop_roi, resolve_roi


def run(config: AppConfig | None = None) -> None:
    app_config = config or AppConfig()

    predictor = create_predictor(
        app_config.predictor_backend,
    )

    decoder = TemporalDecoder(
        min_confidence=app_config.decoder.min_confidence,
        hold_seconds=app_config.decoder.hold_seconds,
        release_seconds=app_config.decoder.release_seconds,
        neutral_labels=app_config.decoder.neutral_labels,
    )

    finalizer = SequenceFinalizer(
        min_confidence=(app_config.finalization.min_confidence),
        background_seconds=(app_config.finalization.background_seconds),
        background_labels=(app_config.finalization.background_labels),
    )

    text_buffer = TextBuffer(separator="")
    last_finalized_text = ""

    speech_engine = create_speech_engine(
        app_config.speech.backend,
    )

    try:
        with Camera(app_config.camera) as camera:
            while True:
                frame = camera.read()

                if app_config.camera.mirror:
                    frame = cv2.flip(frame, 1)

                roi_box = resolve_roi(
                    frame,  # type: ignore
                    app_config.roi,
                )

                roi_image = crop_roi(
                    frame,  # type: ignore
                    roi_box,
                )

                prediction = predictor.predict(
                    roi_image,
                )

                timestamp = monotonic()

                decoder_update = decoder.update(
                    prediction,
                    now=timestamp,
                )

                if decoder_update.accepted_label is not None:
                    text_buffer.append(
                        decoder_update.accepted_label,
                    )

                finalization_update = finalizer.update(
                    prediction,
                    has_content=bool(text_buffer),
                    now=timestamp,
                )

                if finalization_update.finalized:
                    last_finalized_text = text_buffer.text

                    speech_text = last_finalized_text

                    if app_config.speech.lowercase_before_speaking:
                        speech_text = speech_text.lower()

                    speech_engine.speak(speech_text)
                    text_buffer.clear()

                draw_overlay(
                    frame,  # type: ignore
                    roi_box,
                    prediction,
                    decoder_update,
                    finalization_update,
                    text_buffer.text,
                    last_finalized_text,
                )

                cv2.imshow(
                    app_config.window_title,
                    frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in app_config.quit_keys:
                    break

    except CameraError as exc:
        raise SystemExit(f"Camera error: {exc}") from exc

    finally:
        speech_engine.close()
        cv2.destroyAllWindows()
