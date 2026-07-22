import pytest

from sign_translator.decoding.label_handler import (
    LabelAction,
    handle_accepted_label,
)
from sign_translator.decoding.text_buffer import (
    TextBuffer,
)


def test_letter_is_added() -> None:
    buffer = TextBuffer()

    result = handle_accepted_label(
        "a",
        buffer,
    )

    assert buffer.text == "A"
    assert result.action is LabelAction.LETTER_ADDED


def test_space_is_added_once() -> None:
    buffer = TextBuffer()
    buffer.append_letter("A")

    first = handle_accepted_label(
        "space",
        buffer,
    )

    repeated = handle_accepted_label(
        "SPACE",
        buffer,
    )

    assert buffer.text == "A "
    assert first.action is LabelAction.SPACE_ADDED
    assert repeated.action is LabelAction.IGNORED


def test_del_removes_last_character() -> None:
    buffer = TextBuffer()
    buffer.append_letter("A")
    buffer.append_letter("B")

    result = handle_accepted_label(
        "del",
        buffer,
    )

    assert buffer.text == "A"
    assert result.action is LabelAction.CHARACTER_DELETED


def test_nothing_is_ignored() -> None:
    buffer = TextBuffer()
    buffer.append_letter("A")

    result = handle_accepted_label(
        "nothing",
        buffer,
    )

    assert buffer.text == "A"
    assert result.action is LabelAction.IGNORED


def test_unknown_label_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported model label",
    ):
        handle_accepted_label(
            "UNKNOWN",
            TextBuffer(),
        )
