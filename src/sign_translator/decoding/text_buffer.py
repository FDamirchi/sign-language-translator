from dataclasses import dataclass, field


@dataclass(slots=True)
class TextBuffer:
    _characters: list[str] = field(default_factory=list) # type: ignore

    @property
    def text(self) -> str:
        return "".join(self._characters)

    @property
    def finalized_text(self) -> str:
        return self.text.strip()

    def append_letter(self, letter: str) -> None:
        normalized = letter.strip().upper()

        if len(normalized) != 1 or not ("A" <= normalized <= "Z"):
            raise ValueError(f"Expected one English letter, " f"got {letter!r}")

        self._characters.append(normalized)

    def append_space(self) -> bool:
        if not self._characters or self._characters[-1] == " ":
            return False

        self._characters.append(" ")
        return True

    def backspace(self) -> str | None:
        if not self._characters:
            return None

        return self._characters.pop()

    def clear(self) -> None:
        self._characters.clear()

    def __len__(self) -> int:
        return len(self._characters)

    def __bool__(self) -> bool:
        return bool(self._characters)
