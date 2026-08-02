from __future__ import annotations


def parse_inclusive_race_range(
    value: object,
    *,
    argument_name: str = "race range",
) -> tuple[int, int]:
    """Parse one race or an inclusive start/end race range."""
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(f"{argument_name} must contain exactly [<start>, <end>].")
        start = _parse_race_number(value[0], argument_name=argument_name)
        end = _parse_race_number(value[1], argument_name=argument_name)
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            parts = [part.strip() for part in text[1:-1].split(",")]
            if len(parts) != 2:
                raise ValueError(f"{argument_name} must look like '[<start>, <end>]'.")
            start = _parse_race_number(parts[0], argument_name=argument_name)
            end = _parse_race_number(parts[1], argument_name=argument_name)
        elif "-" in text:
            start_text, end_text = text.split("-", maxsplit=1)
            start = _parse_race_number(start_text, argument_name=argument_name)
            end = _parse_race_number(end_text, argument_name=argument_name)
        elif "," in text:
            parts = [part.strip() for part in text.split(",")]
            if len(parts) != 2:
                raise ValueError(f"{argument_name} must contain exactly <start>,<end>.")
            start = _parse_race_number(parts[0], argument_name=argument_name)
            end = _parse_race_number(parts[1], argument_name=argument_name)
        else:
            start = _parse_race_number(text, argument_name=argument_name)
            end = start
    if end < start:
        raise ValueError(f"{argument_name} end must be greater than or equal to start.")
    return start, end


def expand_inclusive_race_range(
    value: object,
    *,
    argument_name: str = "race range",
) -> list[int]:
    start, end = parse_inclusive_race_range(value, argument_name=argument_name)
    return list(range(start, end + 1))


def _parse_race_number(value: object, *, argument_name: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{argument_name} race numbers must be non-empty integers.")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(
            f"{argument_name} race numbers must be integers, got {value!r}."
        ) from exc


__all__ = ["expand_inclusive_race_range", "parse_inclusive_race_range"]
