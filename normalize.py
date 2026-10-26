"""Text normalisation helpers shared by training and inference.

The official comparison (``evaluate.py``) ignores case and repeated spaces, so all
values produced by the pipeline are upper-cased and whitespace-collapsed.  Nothing
else is normalised: punctuation is significant.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional


def norm(value) -> str:
    """Upper-case and collapse whitespace (the official comparison key)."""
    return " ".join(str(value or "").split()).upper()


def digits(value) -> str:
    return re.sub(r"\D", "", str(value or ""))


def tokens(value) -> list[str]:
    return norm(value).split()


def sep_key(value) -> str:
    """Comparison key that ignores hyphen/space differences in compound names."""
    s = norm(value).replace("-", " ").replace("<", " ")
    return " ".join(s.split())


def only_cyrillic(value: str) -> bool:
    return bool(value) and all("А" <= ch <= "Я" or ch == "Ё" or ch == "-" or ch == " "
                               for ch in value)


def levenshtein(a: str, b: str, max_distance: int = 99) -> int:
    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (ca != cb))
        previous = current
    return previous[-1]


def format_date(day: int, month: int, year: int) -> Optional[str]:
    try:
        from datetime import date
        date(year, month, day)
    except ValueError:
        return None
    return f"{day:02d}.{month:02d}.{year:04d}"


def token_confidences(field: Optional[dict]) -> list[float]:
    """Minimum character confidence for every whitespace token of a viz field."""
    if not field:
        return []
    result: list[float] = []
    for line in field.get("lines") or []:
        text = line.get("text") or ""
        conf = line.get("char_conf") or []
        for match in re.finditer(r"\S+", text):
            start, end = match.span()
            if conf and len(conf) >= end:
                result.append(min(conf[start:end]))
            elif conf:
                result.append(min(conf))
            else:
                result.append(0.0)
    return result
