"""Post-process OCR output of Russian internal passports.

Usage::

    python postprocess.py data/test_predictions.jsonl answers.jsonl

For every input document the script builds the ten field candidates (MRZ rules,
reference-book correction and the token statistics learned from the training
labels), scores them with the models stored in ``models/`` and decides whether to
accept the document.  A document is accepted only when all ten fields are jointly
confident, the birth-place toponym gate and the surname gate pass, and the
passport is valid on 2026-10-01.

The script is deterministic, offline and CPU-only.
"""
from __future__ import annotations

import collections
import json
import os
import sys

from normalize import norm
from pipeline import FIELDS, Resources, make_context, select_fields
from scoring import DocScorer
from validity import is_valid_on, parse_date, validity_reason

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def batch_counts(docs: list[dict]) -> dict:
    """Exact OCR value counts over the current batch (labels are not used)."""
    counts = {name: collections.Counter() for name in FIELDS}
    for doc in docs:
        ctx = make_context(doc)
        for name in FIELDS:
            value = ctx.viz_text[name]
            if value:
                counts[name][value] += 1
    return counts


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python postprocess.py <predictions.jsonl> <answers.jsonl>",
              file=sys.stderr)
        return 2
    input_path, output_path = argv[1], argv[2]

    docs = load_jsonl(input_path)
    with open(os.path.join(MODELS, "resources.json"), encoding="utf-8") as handle:
        resources = Resources.from_dict(json.load(handle))
    scorer = DocScorer.load(os.path.join(MODELS, "scorer.json"))
    counts = batch_counts(docs)

    statistics = collections.Counter()
    records = []
    for doc in docs:
        candidates = select_fields(doc, resources, counts)
        probabilities, _, doc_prob, accepted = scorer.score(candidates)
        if not candidates["Birth_place_ru"].gate:
            statistics["rejected_toponym_gate"] += 1
        if not candidates["Last_name_ru"].gate:
            statistics["rejected_surname_gate"] += 1
        birth = parse_date(candidates["Birth_date"].value)
        issue = parse_date(candidates["Issue_date"].value)
        valid = is_valid_on(birth, issue)
        if accepted and not valid:
            accepted = False
            statistics["rejected_invalid_" + validity_reason(birth, issue)] += 1
        if accepted:
            statistics["accepted"] += 1
            fields = {name: norm(candidates[name].value) for name in FIELDS}
            records.append({"doc_id": doc["doc_id"], "decision": "accept",
                            "fields": fields})
        else:
            statistics["rejected"] += 1
            if doc_prob < scorer.doc_threshold:
                statistics["rejected_low_confidence"] += 1
            if any(candidates[name].value is None for name in FIELDS):
                statistics["rejected_missing_field"] += 1
            records.append({"doc_id": doc["doc_id"], "decision": "reject"})

    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"documents: {len(docs)}", file=sys.stderr)
    print(f"accepted:  {statistics['accepted']}", file=sys.stderr)
    print(f"rejected:  {statistics['rejected']}", file=sys.stderr)
    for key, value in sorted(statistics.items()):
        if key not in ("accepted", "rejected"):
            print(f"  {key}: {value}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
