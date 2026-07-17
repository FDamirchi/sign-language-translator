from dataclasses import dataclass, field


@dataclass(slots=True)
class TextBuffer:
    separator: str = " "
    _tokens: list[str] = field(default_factory=list) # type: ignore

    @property
    def text(self) -> str:
        return self.separator.join(self._tokens)

    def append(self, token: str) -> None:
        normalized_token = token.strip()

        if not normalized_token:
            raise ValueError("Cannot append an empty token")

        self._tokens.append(normalized_token)

    def backspace(self) -> str | None:
        if not self._tokens:
            return None
        
        return self._tokens.pop()

    def clear(self) -> None:
        self._tokens.clear()

    def __len__(self) -> int:
        return len(self._tokens)
