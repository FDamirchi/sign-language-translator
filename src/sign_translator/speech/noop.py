class NoOpSpeechEngine:
    def speak(self, text: str) -> None:
        normalized_text = text.strip()

        if not normalized_text:
            return

        print(f"[TTS] {normalized_text}")

    def close(self) -> None:
        pass
