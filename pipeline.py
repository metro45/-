"""Candidate extraction and feature construction.

Every field is handled by a small deterministic rule that returns one candidate
value together with a *reason* code and a set of observable features.  A
per-field logistic model (see ``scoring.py``) then estimates the probability that
the candidate is exactly right; documents are accepted only when every field is
confident and the passport is valid on the reference date.

The rules deliberately prefer *rejecting* to guessing: the MRZ is used only when
its own check digits confirm the value (or when it agrees with the visual zone).
"""
from __future__ import annotations

import collections
import difflib
import math
import re
from dataclasses import dataclass, field as dc_field
from datetime import date
from typing import Optional

from mrz import Mrz, parse_mrz
from normalize import (digits, format_date, levenshtein, norm, only_cyrillic,
                       sep_key, token_confidences)
from validity import REFERENCE_DATE, add_years, parse_date

FIELDS = [
    "Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",
    "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",
    "Issue_organization_ru",
]
NAME_FIELDS = {"Last_name_ru": "surname", "First_name_ru": "first",
               "Middle_name_ru": "middle"}
FREE_FIELDS = ("Birth_place_ru", "Issue_organization_ru")
SEX = {"M": "МУЖ.", "F": "ЖЕН."}

# Abbreviations of settlements stay glued to the following word in the labels
# (``С.СОСНОВКА``), so they must not be split.  Region abbreviations are always
# separated in the labels, therefore a glued ``ОБЛ.ТВЕРСКАЯ`` is split.
GLUED_REGION = re.compile(r"^(ОБЛ|РЕСП|КР|АВТОНОМ|ССР)\.(.{3,})$")

# Abbreviations that the labels write with a full stop.  Settlement
# abbreviations (``ПГТ``, ``ПОС``) are deliberately excluded: the reference
# labels use both spellings, so changing them would introduce errors.
DOT_ANY = {"ОБЛ", "КР", "РЕСП", "ГОР", "ДЕР", "ХУТ", "СТАН", "АВТ", "ОКР",
           "АДМ", "СЛ", "СТ", "ОТД", "ПОС"}
DOT_FIRST = {"С", "Г", "Д", "П", "Х"}
GLUED_SETTLEMENT = re.compile(r"^(ПОС|ХУТ|ДЕР|ГОР|СТАН)([А-ЯЁ].{2,})$")

# Settlement-type markers.  The word immediately after such a marker is the
# settlement name itself; the reference book must never "correct" it, because
# rare but correct village names are often absent from the book while a
# one-letter neighbour is present.
SETTLEMENT_TYPES = {
    "С.", "С", "Г.", "Г", "ГОР.", "ГОР", "ДЕР.", "ДЕР", "Д.", "Д", "П.", "П",
    "ПОС.", "ПОС", "ПГТ.", "ПГТ", "РП", "РП.", "Х.", "Х", "ХУТ.", "ХУТ",
    "СТАН.", "СТАН", "СТ-ЦА", "ДЕРЕВНЯ", "СЕЛО", "ПОСЕЛОК", "ХУТОР",
    "СТАНИЦА", "ГОРОД",
}

# A birth place always contains at least one concrete name; a value made only
# of these administrative words ("ОБЛАСТЬ", "Р-НА") is a truncated OCR line.
GENERIC_PLACE = {
    "ОБЛАСТЬ", "ОБЛ", "РАЙОН", "Р-Н", "Р-НА", "КРАЙ", "КР", "РЕСПУБЛИКА",
    "РЕСП", "ОКРУГ", "ОКР", "АО", "АВТОНОМНЫЙ", "АВТ", "ГОРОД", "ГОР",
    "СЕЛО", "С", "ДЕРЕВНЯ", "ДЕР", "ПОСЕЛОК", "ПОС", "ПГТ", "РП", "ХУТОР",
    "ХУТ", "СТАНИЦА", "СТАН", "СТ-ЦА", "Д", "П", "Х", "Г",
}

# Administrative words that may end a birth place but can never start it: a
# value like "Р-НА БУРЯТИЯ" or "СОВЕТСКОГО Р-НА КРЫМ" is a truncated OCR line
# with the settlement name lost.  In 3000 training labels no correct birth
# place starts with one of these tokens (or with an adjective directly before
# a district marker).
ADMIN_PLACE = {
    "Р-Н", "Р-НА", "РАЙОН", "РАЙОНА", "ОБЛ.", "ОБЛ", "КРАЙ", "КРАЯ", "КР.", "КР",
    "РЕСП.", "РЕСП", "РЕСПУБЛИКА", "РЕСПУБЛИКИ", "АО", "АВТ.", "АВТ", "ОКРУГ",
    "ОКРУГА", "ССР", "СССР",
}
ADJECTIVAL_TAIL = re.compile(
    r"(СКИЙ|СКОГО|СКОЙ|СКАЯ|ЦКИЙ|ЦКОГО|ЦКОЙ|ЦКАЯ|НСКИЙ|НСКОГО|НСКОЙ|НСКАЯ)$")
# An unseen surname ending in "-ШЕНСКИХ" is an unresolved Ш/Щ OCR ambiguity:
# both the visual line and the MRZ can carry the same wrong letter, so the
# document is rejected deliberately instead of relying on an unrelated gate.
AMBIGUOUS_SURNAME = re.compile(r"ШЕНСКИХ$")


@dataclass
class Candidate:
    value: Optional[str]
    reason: str
    feats: dict = dc_field(default_factory=dict)
    gate: bool = True


@dataclass
class Context:
    doc: dict
    mrz: Mrz
    viz_text: dict
    viz_conf: dict
    viz_minchar: dict
    viz_tokens: dict
    viz_token_conf: dict
    global_feats: dict


@dataclass
class Resources:
    ft_vocab: dict
    ft_map: dict
    name_vocab: dict
    name_bigrams: dict
    label_values: dict
    ref_codes: set
    ref_tokens: collections.Counter
    code_orgs: dict

    def as_dict(self) -> dict:
        return {
            "ft_vocab": {k: dict(v) for k, v in self.ft_vocab.items()},
            "ft_map": {k: dict(v) for k, v in self.ft_map.items()},
            "name_vocab": {k: dict(v) for k, v in self.name_vocab.items()},
            "name_bigrams": {k: dict(v) for k, v in self.name_bigrams.items()},
            "label_values": {k: dict(v) for k, v in self.label_values.items()},
            "ref_codes": sorted(self.ref_codes),
            "ref_tokens": dict(self.ref_tokens),
            "code_orgs": {k: dict(v) for k, v in self.code_orgs.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Resources":
        return cls(
            ft_vocab={k: collections.Counter(v) for k, v in data["ft_vocab"].items()},
            ft_map={k: dict(v) for k, v in data["ft_map"].items()},
            name_vocab={k: collections.Counter(v) for k, v in data["name_vocab"].items()},
            name_bigrams={k: collections.Counter(v) for k, v in data["name_bigrams"].items()},
            label_values={k: collections.Counter(v) for k, v in data["label_values"].items()},
            ref_codes=set(data["ref_codes"]),
            ref_tokens=collections.Counter(data["ref_tokens"]),
            code_orgs={k: collections.Counter(v)
                       for k, v in data.get("code_orgs", {}).items()},
        )


def _viz_field(doc: dict, name: str) -> Optional[dict]:
    value = (doc.get("viz") or {}).get(name)
    if not value or not value.get("lines"):
        return None
    return value


def _join_lines(value: Optional[dict]) -> Optional[str]:
    if not value:
        return None
    text = norm(" ".join(line.get("text") or "" for line in value["lines"]))
    return text or None


def make_context(doc: dict) -> Context:
    viz_text, viz_conf, viz_minchar, viz_tokens, viz_token_conf = {}, {}, {}, {}, {}
    for name in FIELDS:
        raw = _viz_field(doc, name)
        viz_text[name] = _join_lines(raw)
        confs = token_confidences(raw)
        viz_tokens[name] = viz_text[name].split() if viz_text[name] else []
        viz_token_conf[name] = confs
        viz_conf[name] = (raw or {}).get("conf")
        viz_minchar[name] = min(confs) if confs else None

    quality = doc.get("quality") or {}
    source = doc.get("source") or "unknown"
    mrz = parse_mrz(doc)
    checks = int(mrz.doc_number_ok) + int(mrz.birth_ok) + int(mrz.composite_ok)
    present = sum(1 for name in FIELDS if viz_text[name])
    global_feats = {
        "src_web": float(source == "web"),
        "src_mobile": float(source == "mobile"),
        "src_mobile_v2": float(source == "mobile_v2"),
        "sharp": float(quality.get("sharpness") or 0) / 200.0,
        "doc_px": float(quality.get("doc_px") or 0) / 2000.0,
        "glare": float(quality.get("glare") or 0),
        "mrz_present": float(mrz.present),
        "mrz_l1min": float(mrz.min_conf1),
        "mrz_l2min": float(mrz.min_conf2),
        "mrz_checks": float(checks),
        "viz_fields": float(present),
        "viz_all10": float(present == len(FIELDS)),
    }
    return Context(doc, mrz, viz_text, viz_conf, viz_minchar, viz_tokens,
                   viz_token_conf, global_feats)


# ---------------------------------------------------------------------------
# resources


def _token_map(pairs: list[tuple[str, str]]) -> dict[str, str]:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for raw, gold in pairs:
        counts[raw][gold] += 1
    result = {}
    for raw, counter in counts.items():
        total = sum(counter.values())
        gold, best = counter.most_common(1)[0]
        if raw != gold and total >= 3 and best / total >= 0.95:
            result[raw] = gold
    return result


def _alignment_pairs(raw: str, gold: str) -> list[tuple[str, str]]:
    raw_tokens, gold_tokens = raw.split(), gold.split()
    matcher = difflib.SequenceMatcher(a=raw_tokens, b=gold_tokens, autojunk=False)
    pairs = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                pairs.append((raw_tokens[i1 + k], gold_tokens[j1 + k]))
        elif tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
            pairs.append((raw_tokens[i1], gold_tokens[j1]))
    return pairs


def build_resources(docs: list[dict], labels: dict[str, dict],
                    ref_codes: Optional[set] = None,
                    ref_tokens: Optional[collections.Counter] = None) -> Resources:
    ft_vocab = {name: collections.Counter() for name in FREE_FIELDS}
    ft_pairs = {name: [] for name in FREE_FIELDS}
    name_vocab = {name: collections.Counter() for name in NAME_FIELDS}
    name_bigrams = {name: collections.Counter() for name in NAME_FIELDS}
    label_values = {name: collections.Counter() for name in FIELDS}
    code_orgs = collections.defaultdict(collections.Counter)

    for doc in docs:
        gold = labels.get(doc["doc_id"])
        if not gold:
            continue
        ctx = make_context(doc)
        code = norm(gold.get("Issue_organisation_code"))
        org = norm(gold.get("Issue_organization_ru"))
        if code and org:
            code_orgs[code][org] += 1
        for name in FIELDS:
            value = norm(gold.get(name))
            if value:
                label_values[name][value] += 1
        for name in FREE_FIELDS:
            raw = ctx.viz_text[name]
            value = norm(gold.get(name))
            if raw:
                ft_pairs[name].extend(_alignment_pairs(raw, value))
            for token in value.split():
                ft_vocab[name][token] += 1
        for name in NAME_FIELDS:
            value = norm(gold.get(name))
            if not value:
                continue
            name_vocab[name][value] += 1
            padded = "^" + value.replace(" ", "") + "$"
            for a, b in zip(padded, padded[1:]):
                name_bigrams[name][a + b] += 1

    return Resources(
        ft_vocab=ft_vocab,
        ft_map={name: _token_map(ft_pairs[name]) for name in FREE_FIELDS},
        name_vocab=name_vocab,
        name_bigrams=name_bigrams,
        label_values=label_values,
        ref_codes=ref_codes or set(),
        ref_tokens=ref_tokens or collections.Counter(),
        code_orgs=dict(code_orgs),
    )


# ---------------------------------------------------------------------------
# structured fields


def _date_from_mrz(text: Optional[str], kind: str) -> Optional[date]:
    if not text or len(text) != 6 or not text.isdigit():
        return None
    yy, month, day = int(text[:2]), int(text[2:4]), int(text[4:])
    year = 2000 + yy if kind == "issue" and yy <= 26 else (1900 + yy if yy > 26 else 2000 + yy)
    return parse_date(f"{day:02d}.{month:02d}.{year:04d}")


def choose_birth(ctx: Context) -> Candidate:
    mrz = ctx.mrz
    value = parse_date(ctx.viz_text["Birth_date"])
    viz_raw = ctx.viz_text["Birth_date"]
    min_conf = ctx.viz_minchar["Birth_date"]
    mrz6 = mrz.birth_yy

    if mrz.line2_ok and mrz.birth_ok and mrz6 and mrz6.isdigit():
        yy, month, day = int(mrz6[:2]), int(mrz6[2:4]), int(mrz6[4:])
        candidates = []
        for year in (1900 + yy, 2000 + yy):
            parsed = parse_date(f"{day:02d}.{month:02d}.{year:04d}")
            if parsed:
                candidates.append(parsed)
        # The visual date resolves the century whenever day/month agree.
        if value and (value.month, value.day) == (month, day) and \
                value.year in (1900 + yy, 2000 + yy):
            return Candidate(format_date(value.day, value.month, value.year),
                             "mrz_check_viz_year",
                             {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
        issue = parse_date(ctx.viz_text["Issue_date"])
        if issue is None and mrz.composite_ok:
            issue = _date_from_mrz(mrz.issue_yy, "issue")
        plausible = [d for d in candidates if 1900 <= d.year <= 2026]
        if issue:
            ages = [d for d in plausible
                    if d <= add_years(issue, -14) and d >= add_years(issue, -100)]
            if ages:
                chosen = max(ages)
                return Candidate(format_date(chosen.day, chosen.month, chosen.year),
                                 "mrz_check_century",
                                 {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
        if plausible:
            chosen = max(plausible)
            if chosen > add_years(REFERENCE_DATE, -14):
                chosen = min(plausible)
            return Candidate(format_date(chosen.day, chosen.month, chosen.year),
                             "mrz_check_century_weak",
                             {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
        return Candidate(None, "mrz_bad_date", {})

    if value and mrz6 and mrz6.isdigit():
        yy, month, day = int(mrz6[:2]), int(mrz6[2:4]), int(mrz6[4:])
        if (value.month, value.day) == (month, day):
            return Candidate(format_date(value.day, value.month, value.year), "agree",
                             {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})

    if value:
        reason = "viz_hi" if (min_conf or 0) >= 0.99 else "viz_lo"
        return Candidate(format_date(value.day, value.month, value.year), reason,
                         {"viz_conf": min_conf or 0})
    return Candidate(None, "missing", {})


def choose_issue(ctx: Context) -> Candidate:
    mrz = ctx.mrz
    value = parse_date(ctx.viz_text["Issue_date"])
    min_conf = ctx.viz_minchar["Issue_date"]

    mrz_date = _date_from_mrz(mrz.issue_yy, "issue")
    if mrz.line2_ok and mrz.composite_ok and mrz_date and \
            date(1997, 1, 1) <= mrz_date <= date(2026, 12, 31):
        return Candidate(format_date(mrz_date.day, mrz_date.month, mrz_date.year),
                         "mrz_composite",
                         {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})

    if value:
        if mrz_date and (value.year, value.month, value.day) == \
                (mrz_date.year, mrz_date.month, mrz_date.day):
            return Candidate(format_date(value.day, value.month, value.year), "agree",
                             {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
        reason = "viz_hi" if (min_conf or 0) >= 0.99 else "viz_lo"
        return Candidate(format_date(value.day, value.month, value.year), reason,
                         {"viz_conf": min_conf or 0})
    return Candidate(None, "missing", {})


def choose_code(ctx: Context) -> Candidate:
    mrz = ctx.mrz
    raw = ctx.viz_text["Issue_organisation_code"]
    min_conf = ctx.viz_minchar["Issue_organisation_code"]
    viz_digits = digits(raw)

    if mrz.line2_ok and mrz.composite_ok and mrz.code:
        return Candidate(mrz.code, "mrz_composite",
                         {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
    if len(viz_digits) == 6:
        code = f"{viz_digits[:3]}-{viz_digits[3:]}"
        if mrz.code and mrz.code == code:
            return Candidate(code, "agree",
                             {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
        reason = "viz_hi" if (min_conf or 0) >= 0.99 else "viz_lo"
        return Candidate(code, reason, {"viz_conf": min_conf or 0})
    return Candidate(None, "missing", {})


def choose_licence(ctx: Context) -> Candidate:
    mrz = ctx.mrz
    raw = ctx.viz_text["Licence_number"]
    min_conf = ctx.viz_minchar["Licence_number"]
    viz_digits = digits(raw)

    if mrz.licence and mrz.doc_number_ok and mrz.composite_ok:
        return Candidate(mrz.licence, "mrz_full",
                         {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
    if len(viz_digits) == 10 and mrz.licence and \
            viz_digits == digits(mrz.licence):
        return Candidate(mrz.licence, "agree",
                         {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
    if mrz.licence and mrz.doc_number_ok:
        return Candidate(mrz.licence, "mrz_doc",
                         {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
    if len(viz_digits) == 10:
        value = f"{viz_digits[:2]} {viz_digits[2:4]} {viz_digits[4:]}"
        reason = "viz_hi" if (min_conf or 0) >= 0.99 else "viz_lo"
        return Candidate(value, reason, {"viz_conf": min_conf or 0})
    return Candidate(None, "missing", {})


def choose_sex(ctx: Context) -> Candidate:
    mrz = ctx.mrz
    raw = ctx.viz_text["Sex_ru"]
    min_conf = ctx.viz_minchar["Sex_ru"]
    if mrz.sex in SEX:
        expected = SEX[mrz.sex]
        if raw == expected:
            return Candidate(expected, "agree",
                             {"viz_conf": min_conf or 0, "mrz_conf": mrz.min_conf2})
        return Candidate(expected, "mrz", {"viz_conf": min_conf or 0,
                                           "mrz_conf": mrz.min_conf2})
    if raw in ("МУЖ.", "ЖЕН."):
        reason = "viz_hi" if (min_conf or 0) >= 0.99 else "viz_lo"
        return Candidate(raw, reason, {"viz_conf": min_conf or 0})
    return Candidate(None, "missing", {})


# ---------------------------------------------------------------------------
# names


def _name_plausibility(value: Optional[str], bigrams: collections.Counter,
                       vocab: collections.Counter) -> dict:
    if not value:
        return {"name_unseen": 0.0, "name_ll": 0.0, "name_unseen_bi": 0.0,
                "name_bad_start": 0.0, "name_double": 0.0, "name_cyr": 0.0}
    padded = "^" + value.replace(" ", "") + "$"
    total = sum(bigrams.values()) + len(bigrams) + 1
    ll, unseen = 0.0, 0
    for a, b in zip(padded, padded[1:]):
        count = bigrams.get(a + b, 0)
        if count == 0:
            unseen += 1
        ll += math.log((count + 0.5) / total)
    return {
        "name_unseen": float(norm(value) not in vocab),
        "name_ll": ll / max(1, len(padded) - 1),
        "name_unseen_bi": float(unseen),
        "name_bad_start": float(value[:1] in ("Ь", "Ъ", "Ы")),
        "name_double": float(bool(re.search(r"(.)\1", value))),
        "name_cyr": float(only_cyrillic(value)),
    }


def _value_features(field: str, value: Optional[str], raw: Optional[str],
                    res: Resources, batch_counts: Optional[dict]) -> dict:
    labels = res.label_values[field]
    value_norm = norm(value)
    raw_norm = norm(raw)
    feats = {
        "val_train": float(labels.get(value_norm, 0) > 0),
        "val_train_freq": float(labels.get(value_norm, 0)),
        "raw_train": float(labels.get(raw_norm, 0) > 0),
        "raw_train_freq": float(labels.get(raw_norm, 0)),
    }
    if batch_counts:
        counts = batch_counts.get(field) or collections.Counter()
        feats.update({
            "batch_val": float(counts.get(value_norm, 0) > 1),
            "batch_val_other": float(max(0, counts.get(value_norm, 0) - 1)),
            "batch_raw": float(counts.get(raw_norm, 0) > 1),
            "batch_raw_other": float(max(0, counts.get(raw_norm, 0) - 1)),
        })
    else:
        feats.update({"batch_val": 0.0, "batch_val_other": 0.0,
                      "batch_raw": 0.0, "batch_raw_other": 0.0})
    return feats


def _surname_gate(name_field: str, value: Optional[str], res: Resources) -> bool:
    """Reject an unattested surname with an unresolved Ш/Щ suffix ambiguity."""
    if name_field != "Last_name_ru" or not value:
        return True
    value = norm(value)
    if not AMBIGUOUS_SURNAME.search(value):
        return True
    return value in res.name_vocab["Last_name_ru"]


def choose_name(ctx: Context, name_field: str, res: Resources,
                batch_counts: Optional[dict] = None) -> Candidate:
    which = NAME_FIELDS[name_field]
    value = ctx.viz_text[name_field]
    mrz_value = ctx.mrz.name(which)
    min_conf = ctx.viz_minchar[name_field]
    mrz_conf = ctx.mrz.min_conf1
    feats = {"viz_conf": min_conf or 0, "mrz_conf": mrz_conf}
    bigrams = res.name_bigrams[name_field]

    def enrich(chosen: Optional[str]) -> dict:
        extra = _name_plausibility(chosen, bigrams, res.name_vocab[name_field])
        extra.update(_value_features(name_field, chosen, value, res, batch_counts))
        return {**feats, **extra}

    def cand(chosen: Optional[str], reason: str, features: dict) -> Candidate:
        return Candidate(chosen, reason, features,
                         _surname_gate(name_field, chosen, res))

    if not value and not mrz_value:
        empty = "" if name_field == "Middle_name_ru" else None
        return cand(empty, "missing", enrich(empty))
    if not value:
        return cand(mrz_value, "mrz_only", enrich(mrz_value))
    if not mrz_value:
        return cand(value, "viz_only", enrich(value))

    if norm(value) == norm(mrz_value):
        return cand(value, "agree", enrich(value))
    if sep_key(value) == sep_key(mrz_value):
        return cand(value, "agree_sep", enrich(value))

    distance = levenshtein(norm(value), norm(mrz_value))
    if distance <= 1:
        if (min_conf or 0) >= 0.99 and mrz_conf < 0.98:
            return cand(value, "viz_fix", enrich(value))
        if mrz_conf >= 0.99 and (min_conf or 0) < 0.98:
            return cand(mrz_value, "mrz_fix", enrich(mrz_value))
    if mrz_value.startswith(value) and len(mrz_value) - len(value) <= 2 and (min_conf or 0) >= 0.99:
        return cand(value, "viz_prefix", enrich(value))
    if value.startswith(mrz_value) and len(value) - len(mrz_value) <= 2 and mrz_conf >= 0.99:
        return cand(mrz_value, "mrz_prefix", enrich(mrz_value))
    return cand(None, "disagree", feats)


# ---------------------------------------------------------------------------
# free text


def _free_tokens(raw: str, mapping: dict[str, str],
                 confs: list[float]) -> tuple[list[str], list[float], int, int]:
    out, out_conf, mapped, split = [], [], 0, 0
    for index, token in enumerate(raw.split()):
        conf = confs[index] if index < len(confs) else (min(confs) if confs else 0.0)
        match = GLUED_REGION.match(token)
        if match:
            pieces = [match.group(1) + ".", match.group(2)]
            split += 1
        else:
            pieces = [token]
        for piece in pieces:
            replacement = mapping.get(piece, piece)
            if replacement != piece:
                mapped += 1
            out.append(replacement)
            out_conf.append(conf)
    return out, out_conf, mapped, split


def _nearest(token: str, vocab: collections.Counter, maxd: int,
             minfreq: int) -> Optional[str]:
    best = []
    for word, count in vocab.items():
        if count < minfreq or abs(len(word) - len(token)) > maxd:
            continue
        distance = levenshtein(token, word, maxd)
        if distance <= maxd:
            best.append((distance, -count, word))
    if not best:
        return None
    best.sort()
    distance = best[0][0]
    same = [item for item in best if item[0] == distance]
    if len(same) != 1:
        return None
    return same[0][2]


def _suffix_only(a: str, b: str) -> bool:
    """True when two words differ only near their endings (an inflection)."""
    prefix = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        prefix += 1
    return prefix >= min(len(a), len(b)) - 2


_ALPHABET = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"


def _variants1(word: str) -> set[str]:
    """All strings within one Levenshtein edit of ``word``."""
    result = set()
    for index in range(len(word) + 1):
        for char in _ALPHABET:
            result.add(word[:index] + char + word[index:])
    for index in range(len(word)):
        result.add(word[:index] + word[index + 1:])
        for char in _ALPHABET:
            result.add(word[:index] + char + word[index + 1:])
        if index + 1 < len(word):
            result.add(word[:index] + word[index + 1] + word[index] + word[index + 2:])
    return result


def _yo_only(a: str, b: str) -> bool:
    return a.replace("Ё", "Е") == b.replace("Ё", "Е")


def _fix_punctuation(tokens: list[str], field: str) -> tuple[list[str], int]:
    """Restore the full stops that the labels consistently use after the
    abbreviations of settlement/region types."""
    out, fixed = [], 0
    for index, token in enumerate(tokens):
        # OCR occasionally doubles the full stop ("ОБЛ.."); the labels never do.
        original = token
        while token.endswith(".."):
            token = token[:-1]
        new = token
        if field == "Birth_place_ru":
            if not token.endswith(".") and token in DOT_ANY:
                new = token + "."
            elif index == 0 and token in DOT_FIRST:
                new = token + "."
            else:
                match = GLUED_SETTLEMENT.match(token)
                if match and match.group(2)[0] not in "ЕЁИОУЫЭЮЯ":
                    new = match.group(1) + "." + match.group(2)
        elif not token.endswith(".") and token in ("Г", "ГОР"):
            new = token + "."
        if new != original:
            fixed += 1
        out.append(new)
    return out, fixed


def _ref_correct(tokens: list[str], field: str, vocab: collections.Counter,
                 ref_tokens: collections.Counter, min_freq: int,
                 skip_index: Optional[int] = None) -> tuple[list[str], int, int]:
    """Fallback correction against the subdivision reference book.

    The reference contains many rare place names that never occur in the
    training labels.  A token is replaced only when exactly one reference word
    is one edit away and the word is attested at least ``min_freq`` times.
    ``skip_index`` protects the settlement name itself: the book is helpful for
    region/district adjectives but harmful for rare villages.
    """
    out, corrected, min_seen = [], 0, 0
    for index, token in enumerate(tokens):
        if index == skip_index:
            out.append(token)
            continue
        if token in vocab or token in ref_tokens or len(token) < 5 or \
                not re.fullmatch(r"[А-ЯЁ-]+", token):
            out.append(token)
            continue
        hits = [word for word in _variants1(token)
                if word in ref_tokens and ref_tokens[word] >= min_freq]
        if len(hits) == 1:
            replacement = hits[0]
            if not _suffix_only(token, replacement) and not _yo_only(token, replacement):
                corrected += 1
                min_seen = ref_tokens[replacement] if not min_seen else min(min_seen,
                                                                           ref_tokens[replacement])
                out.append(replacement)
                continue
        out.append(token)
    return out, corrected, min_seen


def _settlement_skip(tokens: list[str]) -> Optional[int]:
    """Index of the settlement name that must not be reference-corrected."""
    for index, token in enumerate(tokens):
        if token in SETTLEMENT_TYPES:
            return index + 1 if index + 1 < len(tokens) else None
    return 0 if tokens else None


def _place_complete(tokens: list[str]) -> bool:
    """A birth place must not start with an administrative tail."""
    if not tokens:
        return False
    if tokens[0] in ADMIN_PLACE:
        return False
    if (len(tokens) > 1 and tokens[1] in ADMIN_PLACE
            and ADJECTIVAL_TAIL.search(tokens[0])):
        return False
    return True


def _code_lookup(value: str, code: Optional[str],
                 code_orgs: dict) -> tuple[str, dict]:
    """Use the issuing-authority code to repair a nearly matching name.

    ``Issue_organization_ru`` for a given code has a small number of attested
    spellings.  If the OCR value is extremely close to one of them, it is
    almost certainly the same organisation with a single recognition error.
    """
    names = code_orgs.get(norm(code)) if code else None
    if not names:
        return value, {"lookup_known": 0.0, "lookup_n": 0.0,
                       "lookup_dist": 0.0, "lookup_freq": 0.0}
    best = None
    for name, count in names.items():
        ratio = difflib.SequenceMatcher(a=value, b=name, autojunk=False).ratio()
        candidate = (1.0 - ratio, -count, name)
        if best is None or candidate < best:
            best = candidate
    distance, neg_count, name = best
    feats = {"lookup_known": 1.0, "lookup_n": float(len(names)),
             "lookup_dist": float(distance), "lookup_freq": float(-neg_count)}
    if name != value and distance <= 0.02:
        return name, feats
    return value, feats



def _correct_unknown(tokens: list[str], field: str,
                     vocab: collections.Counter,
                     ref_tokens: collections.Counter) -> tuple[list[str], int]:
    """Correct an unknown token only when a single frequent vocabulary word is
    within a small edit distance.  Known tokens and words attested in the
    reference book are never touched, and inflection-only differences are left
    alone, so valid rare toponyms survive."""
    out, corrected = [], 0
    for token in tokens:
        if token in vocab:
            out.append(token)
            continue
        if len(token) < 6 or not re.fullmatch(r"[А-ЯЁ-]+", token):
            out.append(token)
            continue
        if token in ref_tokens:
            out.append(token)
            continue
        maxd = 2 if len(token) >= 10 else 1
        minfreq = 50 if field == "Birth_place_ru" else 30
        replacement = _nearest(token, vocab, maxd, minfreq)
        if replacement and not _suffix_only(token, replacement):
            corrected += 1
            out.append(replacement)
        else:
            out.append(token)
    return out, corrected


def choose_free_text(ctx: Context, name_field: str, res: Resources,
                     batch_counts: Optional[dict] = None) -> Candidate:
    raw = ctx.viz_text[name_field]
    if not raw:
        return Candidate(None, "missing", {})
    tokens, confs, mapped, split = _free_tokens(
        raw, res.ft_map[name_field], ctx.viz_token_conf[name_field])
    vocab = res.ft_vocab[name_field]
    unknown_before = sum(token not in vocab for token in tokens)
    tokens, corrected = _correct_unknown(tokens, name_field, vocab, res.ref_tokens)
    tokens, punct_fixed = _fix_punctuation(tokens, name_field)
    min_ref_freq = 10 if name_field == "Birth_place_ru" else 1
    skip = _settlement_skip(tokens) if name_field == "Birth_place_ru" else None
    tokens, ref_fixed, ref_min = _ref_correct(
        tokens, name_field, vocab, res.ref_tokens, min_ref_freq, skip)
    value = " ".join(tokens)

    lookup_feats = {"lookup_known": 0.0, "lookup_n": 0.0,
                    "lookup_dist": 0.0, "lookup_freq": 0.0}
    if name_field == "Issue_organization_ru":
        code = choose_code(ctx)
        if code.reason in ("mrz_composite", "agree") and code.value:
            value, lookup_feats = _code_lookup(value, code.value, res.code_orgs)
            tokens = value.split()

    known = [token for token in tokens if token in vocab]
    unknown = [token for token in tokens if token not in vocab]
    short = [token for token in tokens if len(token) <= 2 and token not in (".", "-")]
    ref = res.ref_tokens
    ref_unknown = [token for token in tokens
                   if token not in ref and token not in vocab and len(token) >= 4]

    # Toponym hard gate: every word of the birth place must be attested either
    # in the gold-label vocabulary or in the subdivision reference book, and the
    # value must contain a concrete (non-generic) name.  The reference book is
    # needed because many correct rare villages never occur in the 3000 labels;
    # the settlement name itself is protected from reference *correction* above,
    # so a ref-attested word no longer means "a corrected wrong village".
    gate = True
    if name_field == "Birth_place_ru":
        gate = (bool(tokens)
                and _place_complete(tokens)
                and all(token in vocab or token in res.ref_tokens
                        for token in tokens)
                and any(token.rstrip(".") not in GENERIC_PLACE for token in tokens))
    feats = {
        "n_tok": float(len(tokens)),
        "n_known": float(len(known)),
        "known_frac": len(known) / max(1, len(tokens)),
        "n_unknown": float(len(unknown)),
        "unk_frac": len(unknown) / max(1, len(tokens)),
        "n_mapped": float(mapped),
        "n_split": float(split),
        "n_corrected": float(corrected),
        "n_punct_fixed": float(punct_fixed),
        "n_ref_fixed": float(ref_fixed),
        "ref_fix_min_freq": float(ref_min),
        "n_unknown_raw": float(unknown_before),
        "n_short": float(len(short)),
        "n_digit": float(sum(any(ch.isdigit() for ch in t) for t in tokens)),
        "n_chars": float(len(value)),
        "n_lines": float(len((ctx.doc.get("viz") or {}).get(name_field, {}).get("lines") or [])),
        "viz_conf": ctx.viz_conf[name_field] or 0.0,
        "min_char": min(confs) if confs else 0.0,
        "mean_char": sum(confs) / max(1, len(confs)),
        "n_conf_lt99": float(sum(c < 0.99 for c in confs)),
        "n_conf_lt90": float(sum(c < 0.90 for c in confs)),
        "latin": float(bool(re.search(r"[A-Z]", value))),
        "ends_dot": float(value.endswith(".")),
        "n_ref_unknown": float(len(ref_unknown)),
        "n_ref_unknown_frac": len(ref_unknown) / max(1, len(tokens)),
        "duplicate": float(any(a == b for a, b in zip(tokens, tokens[1:]))),
    }
    feats.update(lookup_feats)
    feats.update(_value_features(name_field, value, raw, res, batch_counts))
    return Candidate(value, "ft", feats, gate)


def select_fields(doc: dict, res: Resources,
                  batch_counts: Optional[dict] = None) -> dict[str, Candidate]:
    ctx = make_context(doc)
    result = {
        "Last_name_ru": choose_name(ctx, "Last_name_ru", res, batch_counts),
        "First_name_ru": choose_name(ctx, "First_name_ru", res, batch_counts),
        "Middle_name_ru": choose_name(ctx, "Middle_name_ru", res, batch_counts),
        "Sex_ru": choose_sex(ctx),
        "Birth_date": choose_birth(ctx),
        "Birth_place_ru": choose_free_text(ctx, "Birth_place_ru", res, batch_counts),
        "Issue_date": choose_issue(ctx),
        "Issue_organisation_code": choose_code(ctx),
        "Licence_number": choose_licence(ctx),
        "Issue_organization_ru": choose_free_text(ctx, "Issue_organization_ru", res, batch_counts),
    }
    for name, candidate in result.items():
        candidate.feats = {**ctx.global_feats, **candidate.feats,
                           "reason": candidate.reason}
    return result


def feature_dict(candidate: Candidate) -> dict[str, float]:
    """One-hot encoded version of a candidate's features (for DictVectorizer)."""
    out = {}
    for key, value in candidate.feats.items():
        if key == "reason":
            out[f"reason={value}"] = 1.0
        else:
            out[key] = float(value)
    return out
