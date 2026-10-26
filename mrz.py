"""Parsing and validation of the machine-readable zone of a Russian passport.

The MRZ layout was reconstructed empirically from the training data (see REPORT.md).
Only the parts needed for post-processing are decoded here; every decoded value is
accompanied by the corresponding check-digit flag so the caller can decide how much
to trust it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Deterministic MRZ transliteration used by the generator of this dataset.
# It is *not* the ICAO transliteration: e.g. Ж -> J, Ч -> 3, Ш -> 4, Щ -> W.
CYR_TO_MRZ = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "2",
    "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K", "Л": "L", "М": "M",
    "Н": "N", "О": "O", "П": "P", "Р": "R", "С": "S", "Т": "T", "У": "U",
    "Ф": "F", "Х": "H", "Ц": "C", "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X",
    "Ы": "Y", "Ь": "9", "Э": "6", "Ю": "7", "Я": "8", "-": "<", " ": "<",
}
MRZ_TO_CYR = {v: k for k, v in CYR_TO_MRZ.items() if k not in "- "}
MRZ_TO_CYR["<"] = "-"


def check_digit(text: str) -> int:
    """ICAO 7-3-1 check digit."""
    weights = (7, 3, 1)
    total = 0
    for i, ch in enumerate(text):
        if ch.isdigit():
            value = ord(ch) - 48
        elif "A" <= ch <= "Z":
            value = ord(ch) - 55
        else:
            value = 0
        total += value * weights[i % 3]
    return total % 10


def _decode(text: str) -> str:
    return "".join(MRZ_TO_CYR.get(ch, ch) for ch in text)


def decode_name_line(line1: str) -> Optional[dict]:
    """Decode ``PNRUS<SURNAME><<GIVEN<MIDDLE<<<`` into Cyrillic parts.

    Returns ``None`` when the line does not look like a name line.  A ``<`` inside a
    surname/given-name component is a hyphen; components of a compound given name
    are returned separately so the caller can choose the separator.
    """
    marker = line1.find("RUS")
    if marker < 0:
        return None
    body = line1[marker + 3:]
    if "<<" not in body:
        return None
    surname, given = body.split("<<", 1)
    # Everything after a double '<' is padding or a stray misread character.
    given = given.split("<<", 1)[0].rstrip("<")
    parts = [p for p in given.split("<") if p]
    if not parts:
        return None
    return {
        "surname": _decode(surname).replace("<", "-"),
        "first": _decode(parts[0]),
        "middle_parts": [_decode(p).replace("<", "-") for p in parts[1:]],
    }


@dataclass
class Mrz:
    present: bool = False
    line1: str = ""
    line2: str = ""
    min_conf1: float = 0.0
    min_conf2: float = 0.0

    # line 2
    doc_number_ok: bool = False       # check digit over series(3) + number(6)
    birth_ok: bool = False            # check digit over YYMMDD
    composite_ok: bool = False        # check digit over series[3] + issue + code

    licence: Optional[str] = None      # "61 24 283968"
    birth_yy: Optional[str] = None     # YYMMDD
    sex: Optional[str] = None          # "M" / "F"
    issue_yy: Optional[str] = None     # YYMMDD
    code: Optional[str] = None         # "620-004"

    # line 1
    surname: Optional[str] = None
    first: Optional[str] = None
    middle_parts: list[str] = field(default_factory=list)

    @property
    def line2_ok(self) -> bool:
        return self.present and len(self.line2) >= 44

    @property
    def name_ok(self) -> bool:
        return self.surname is not None and self.first is not None

    def name(self, which: str) -> Optional[str]:
        if which == "surname":
            return self.surname
        if which == "first":
            return self.first
        if which == "middle":
            if not self.middle_parts:
                return None
            # A single component is the normal case.  Multiple components are
            # typically "ОГЛЫ"/"КЫЗЫ"; the separator is resolved by the caller.
            return " ".join(self.middle_parts)
        return None


def parse_mrz(doc: dict) -> Mrz:
    mrz = Mrz()
    raw = doc.get("mrz")
    if not raw:
        return mrz
    lines = raw.get("lines") or []
    if not lines:
        return mrz
    mrz.present = True
    mrz.line1 = (lines[0].get("text") or "") if len(lines) > 0 else ""
    mrz.line2 = (lines[1].get("text") or "") if len(lines) > 1 else ""
    conf1 = lines[0].get("char_conf") or []
    conf2 = (lines[1].get("char_conf") or []) if len(lines) > 1 else []
    mrz.min_conf1 = min(conf1) if conf1 else 0.0
    mrz.min_conf2 = min(conf2) if conf2 else 0.0

    names = decode_name_line(mrz.line1) if mrz.line1 else None
    if names:
        mrz.surname = names["surname"]
        mrz.first = names["first"]
        mrz.middle_parts = names["middle_parts"]

    l2 = mrz.line2
    if len(l2) < 44:
        return mrz

    doc9 = l2[0:9]
    if l2[9].isdigit() and doc9.isdigit():
        mrz.doc_number_ok = check_digit(doc9) == int(l2[9])
    series = l2[0:3] + l2[28]
    if len(series) == 4 and series.isdigit() and l2[3:9].isdigit():
        mrz.licence = f"{series[:2]} {series[2:]} {l2[3:9]}"

    mrz.birth_yy = l2[13:19]
    if l2[19].isdigit() and mrz.birth_yy.isdigit():
        mrz.birth_ok = check_digit(mrz.birth_yy) == int(l2[19])

    mrz.sex = l2[20] if l2[20] in ("M", "F") else None
    mrz.issue_yy = l2[29:35]
    code6 = l2[35:41]
    if code6.isdigit():
        mrz.code = f"{code6[:3]}-{code6[3:]}"

    composite = l2[28:41]
    if l2[42].isdigit():
        mrz.composite_ok = check_digit(composite) == int(l2[42])
    return mrz
