from sign_translator.decoding.text_buffer import (
    TextBuffer,
)


def test_appends_letters_without_separator() -> None:
    buffer = TextBuffer()

    buffer.append_letter("C")
    buffer.append_letter("A")
    buffer.append_letter("T")

    assert buffer.text == "CAT"
    assert len(buffer) == 3


def test_letters_are_normalized_to_uppercase() -> None:
    buffer = TextBuffer()

    buffer.append_letter("a")
    buffer.append_letter("b")

    assert buffer.text == "AB"


def test_space_is_not_added_at_start_or_repeated() -> None:
    buffer = TextBuffer()

    assert buffer.append_space() is False

    buffer.append_letter("A")

    assert buffer.append_space() is True
    assert buffer.append_space() is False
    assert buffer.text == "A "


def test_backspace_removes_last_character() -> None:
    buffer = TextBuffer()

    buffer.append_letter("C")
    buffer.append_space()
    buffer.append_letter("A")

    removed = buffer.backspace()

    assert removed == "A"
    assert buffer.text == "C "


def test_finalized_text_strips_outer_spaces() -> None:
    buffer = TextBuffer()

    buffer.append_letter("A")
    buffer.append_space()

    assert buffer.text == "A "
    assert buffer.finalized_text == "A"


def test_clear_removes_all_characters() -> None:
    buffer = TextBuffer()

    buffer.append_letter("A")
    buffer.clear()

    assert buffer.text == ""
    assert len(buffer) == 0
    assert bool(buffer) is False
