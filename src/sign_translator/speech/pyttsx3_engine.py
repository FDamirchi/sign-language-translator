from queue import Queue
from threading import Event, Thread
from types import ModuleType


class Pyttsx3SpeechEngine:
    def __init__(self) -> None:
        self._queue: Queue[str | None] = Queue()
        self._ready = Event()
        self._startup_error: BaseException | None = None
        self._closed = False
        self._pyttsx3: ModuleType | None = None

        try:
            import pyttsx3 # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyttsx3 is not installed.\nInstall the TTS optional dependencies."
            ) from exc

        self._pyttsx3 = pyttsx3

        self._worker = Thread(
            target=self._run_worker,
            name="tts-worker",
            daemon=True,
        )
        self._worker.start()

        self._ready.wait()

        if self._startup_error is not None:
            raise RuntimeError(
                "Could not initialize the TTS engine"
            ) from self._startup_error

    def speak(self, text: str) -> None:
        if self._closed:
            raise RuntimeError("Speech engine is already closed")

        normalized_text = text.strip()

        if not normalized_text:
            return

        self._queue.put(normalized_text)

    def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        self._queue.put(None)
        self._worker.join()

    def _run_worker(self) -> None:
        engine = None

        try:
            if self._pyttsx3 is None:
                raise RuntimeError("pyttsx3 module is unavailable")

            engine = self._pyttsx3.init()
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return

        self._ready.set()

        while True:
            text = self._queue.get()

            if text is None:
                engine.stop()
                return

            engine.say(text)
            engine.runAndWait()
