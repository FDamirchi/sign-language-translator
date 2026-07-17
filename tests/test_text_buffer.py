from sign_translator.decoding.text_buffer import TextBuffer


def test_appends_tokens() -> None:
    buffer = TextBuffer(separator=" ")

    buffer.append("A")
    buffer.append("B")

    assert buffer.text == "A B"
    assert len(buffer) == 2


def test_backspace_removes_last_token() -> None:
    buffer = TextBuffer(separator=" ")

    buffer.append("A")
    buffer.append("B")

    removed = buffer.backspace()

    assert removed == "B"
    assert buffer.text == "A"


def test_clear_removes_all_tokens() -> None:
    buffer = TextBuffer()

    buffer.append("A")
    buffer.clear()

    assert buffer.text == ""
    assert len(buffer) == 0
