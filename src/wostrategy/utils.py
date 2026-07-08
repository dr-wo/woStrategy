from __future__ import annotations

from collections.abc import Iterable


def parse_inclusive_race_range(
    value: object,
    *,
    argument_name: str = "race range",
) -> tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{argument_name} must contain exactly [<start>, <end>]."
            )
        start = _parse_race_number(value[0], argument_name=argument_name)
        end = _parse_race_number(value[1], argument_name=argument_name)
    else:
        text = str(value).strip()
        if text.startswith("[") and text.endswith("]"):
            parts = [part.strip() for part in text[1:-1].split(",")]
            if len(parts) != 2:
                raise ValueError(
                    f"{argument_name} must look like '[<start>, <end>]'."
                )
            start = _parse_race_number(parts[0], argument_name=argument_name)
            end = _parse_race_number(parts[1], argument_name=argument_name)
        elif "-" in text:
            start_text, end_text = text.split("-", maxsplit=1)
            start = _parse_race_number(start_text, argument_name=argument_name)
            end = _parse_race_number(end_text, argument_name=argument_name)
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


def wcc_leading_team(*, year: int, end_race: int | str) -> str:
    """Return the constructor leading the WCC after the requested race."""
    try:
        from fastf1.ergast import Ergast

        standings = Ergast().get_constructor_standings(
            season=year,
            round=end_race,
            standings_position=1,
        )
        if not standings.content:
            raise ValueError("empty constructor standings response")
        leader = standings.content[0].iloc[0]
        return str(leader["constructorName"])
    except Exception as exc:
        raise RuntimeError(
            "Could not determine the default reference team from WCC standings. "
            "Configure a reference team explicitly."
        ) from exc


def reference_team_or_wcc_leader(
    *,
    year: int,
    end_race: int | str,
    reference_team: str | None,
) -> str:
    if reference_team is not None:
        return reference_team
    return wcc_leading_team(year=year, end_race=end_race)


def match_team_name(reference_team: str, available_teams: Iterable[object]) -> str:
    teams = [str(team) for team in available_teams]
    if reference_team in teams:
        return reference_team

    lower_match = {team.lower(): team for team in teams}
    if reference_team.lower() in lower_match:
        return lower_match[reference_team.lower()]

    normalized_match = {_normalize_team_name(team): team for team in teams}
    normalized_reference = _normalize_team_name(reference_team)
    if normalized_reference in normalized_match:
        return normalized_match[normalized_reference]

    alias_reference = _team_alias(normalized_reference)
    for team in teams:
        if _team_alias(_normalize_team_name(team)) == alias_reference:
            return team

    available = ", ".join(sorted(teams))
    raise ValueError(
        f"Reference team {reference_team!r} was not found. "
        f"Available teams: {available}"
    )


def _normalize_team_name(team: str) -> str:
    normalized = team.lower().replace(" f1 team", "").replace(" racing", "")
    return " ".join(normalized.split())


def _team_alias(normalized_team: str) -> str:
    aliases = {
        "red bull": "red bull",
        "red bull racing": "red bull",
        "rb": "racing bulls",
        "visa cash app rb": "racing bulls",
        "haas": "haas",
        "haas f1 team": "haas",
    }
    return aliases.get(normalized_team, normalized_team)


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


__all__ = [
    "expand_inclusive_race_range",
    "match_team_name",
    "parse_inclusive_race_range",
    "reference_team_or_wcc_leader",
    "wcc_leading_team",
]
