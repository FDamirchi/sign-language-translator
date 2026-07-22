from sign_translator.decoding.text_buffer import TextBuffer


def test_appends_letters_without_separator() -> None:
    buffer = TextBuffer()

    buffer.append("C")
    buffer.append("A")
    buffer.append("T")

    assert buffer.text == "CAT"
    assert len(buffer) == 3


def test_letters_are_normalized_to_uppercase() -> None:
    buffer = TextBuffer()

    buffer.append("a")
    buffer.append("b")

    assert buffer.text == "AB"


def test_backspace_removes_last_letter() -> None:
    buffer = TextBuffer()

    buffer.append("C")
    buffer.append("A")
    buffer.append("T")

    removed = buffer.backspace()

    assert removed == "T"
    assert buffer.text == "CA"


def test_clear_removes_all_letters() -> None:
    buffer = TextBuffer()

    buffer.append("A")
    buffer.clear()

    assert buffer.text == ""
    assert len(buffer) == 0
    assert bool(buffer) is False
