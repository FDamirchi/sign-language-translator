from typing import Protocol


class SpeechEngine(Protocol):
    def speak(self, text: str) -> None:
        """Speak or proccess the provided text."""
        ...

    def close(self) -> None:
        """Release resources taken by the speech engine."""
        ...
