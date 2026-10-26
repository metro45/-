"""Passport validity on a fixed reference date.

Russian internal passports are tied to age milestones: a passport issued before the
holder's 20th birthday becomes invalid at 20, and one issued before the 45th birthday
becomes invalid at 45.  The reference date for this task is 2026-10-01.
"""
from __future__ import annotations

from datetime import date
import re
from typing import Optional

REFERENCE_DATE = date(2026, 10, 1)
# Russian internal passports in the dataset were issued no earlier than 1997.
MIN_ISSUE_DATE = date(1997, 1, 1)


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    s = " ".join(str(value).split()).upper()
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})\.(\d{2}|\d{4})", s)
    if not m:
        return None
    day, month, year = (int(x) for x in m.groups())
    if year < 100:
        year += 2000 if year <= 26 else 1900
    try:
        return date(year, month, day)
    except ValueError:
        return None


def add_years(value: date, years: int) -> date:
    """Add years, mapping 29 February to 28 February in a non-leap year."""
    try:
        return value.replace(year=value.year + years)
    except ValueError:
        return date(value.year + years, 2, 28)


def expand_two_digit_year(yy: int, *, kind: str) -> Optional[int]:
    """Choose a century for a two-digit year.

    ``kind='issue'``: passport issue dates are in 1997..2026.
    ``kind='birth'``: the caller must additionally check the age; this helper only
    returns the usual candidate.
    """
    if kind == "issue":
        year = 2000 + yy if yy <= 26 else 1900 + yy
        return year
    if kind == "birth":
        return 2000 + yy if yy <= 26 else 1900 + yy
    return None


def birth_year_candidates(yy: int) -> list[int]:
    return [1900 + yy, 2000 + yy]


def is_valid_on(birth: Optional[date], issue: Optional[date],
                on: date = REFERENCE_DATE) -> bool:
    if birth is None or issue is None:
        return False
    if issue < add_years(birth, 14):
        return False
    if issue > on:
        return False
    if issue < MIN_ISSUE_DATE:
        return False
    if on >= add_years(birth, 45):
        return issue >= add_years(birth, 45)
    if on >= add_years(birth, 20):
        return issue >= add_years(birth, 20)
    return True


def validity_reason(birth: Optional[date], issue: Optional[date],
                    on: date = REFERENCE_DATE) -> str:
    if birth is None or issue is None:
        return "unknown_dates"
    if issue < add_years(birth, 14):
        return "issued_before_14"
    if issue > on:
        return "issue_in_future"
    if issue < MIN_ISSUE_DATE:
        return "issue_too_early"
    if on >= add_years(birth, 45) and issue < add_years(birth, 45):
        return "not_renewed_45"
    if on >= add_years(birth, 20) and issue < add_years(birth, 20):
        return "not_renewed_20"
    return "valid"
