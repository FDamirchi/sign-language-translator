from dataclasses import dataclass
from enum import Enum, auto

from sign_translator.decoding.text_buffer import TextBuffer
from sign_translator.labels import (
    DELETE_LABEL,
    NOTHING_LABEL,
    SPACE_LABEL,
    is_letter_label,
    normalize_label,
)


class LabelAction(Enum):
    LETTER_ADDED = auto()
    SPACE_ADDED = auto()
    CHARACTER_DELETED = auto()
    IGNORED = auto()


@dataclass(frozen=True, slots=True)
class LabelHandlingResult:
    label: str
    action: LabelAction


def handle_accepted_label(
    label: str,
    buffer: TextBuffer,
) -> LabelHandlingResult:
    normalized = normalize_label(label)

    if is_letter_label(normalized):
        buffer.append_letter(normalized)

        return LabelHandlingResult(
            label=normalized,
            action=LabelAction.LETTER_ADDED,
        )

    if normalized == SPACE_LABEL:
        added = buffer.append_space()

        return LabelHandlingResult(
            label=normalized,
            action=(LabelAction.SPACE_ADDED if added else LabelAction.IGNORED),
        )

    if normalized == DELETE_LABEL:
        removed = buffer.backspace()

        return LabelHandlingResult(
            label=normalized,
            action=(
                LabelAction.CHARACTER_DELETED
                if removed is not None
                else LabelAction.IGNORED
            ),
        )

    if normalized == NOTHING_LABEL:
        return LabelHandlingResult(
            label=normalized,
            action=LabelAction.IGNORED,
        )

    raise ValueError(f"Unsupported model label: {label!r}")
