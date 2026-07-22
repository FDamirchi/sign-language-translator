from string import ascii_uppercase


LETTER_LABELS: tuple[str, ...] = tuple(ascii_uppercase)

DELETE_LABEL = "DEL"
NOTHING_LABEL = "NOTHING"
SPACE_LABEL = "SPACE"

COMMAND_LABELS = frozenset(
    {
        DELETE_LABEL,
        NOTHING_LABEL,
        SPACE_LABEL,
    }
)

SUPPORTED_LABELS = frozenset(LETTER_LABELS) | COMMAND_LABELS

MODEL_OUTPUT_LABELS: tuple[str, ...] = (
    *LETTER_LABELS,
    DELETE_LABEL,
    NOTHING_LABEL,
    SPACE_LABEL,
)


def normalize_label(label: str) -> str:
    normalized = label.strip().upper()

    if not normalized:
        raise ValueError("Label cannot be empty!")

    return normalized


def is_letter_label(label: str) -> bool:
    return normalize_label(label) in LETTER_LABELS
