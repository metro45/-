"""Train the per-field correctness models and calibrate the document rule.

Usage::

    python train.py

The script performs a 5-fold out-of-fold evaluation.  Vocabularies, token maps,
the code -> organisation map and name statistics are rebuilt inside every fold;
the correctness models are trained on the other folds.  The document-level
threshold is selected on those OOF predictions.  Finally the models are refitted
on all training data and written to ``models/scorer.json`` together with
``models/resources.json``.

The accept rule is deliberately simple and calibrated on OOF data: accept when
the product of the ten per-field probabilities is at least ``DOC_THRESHOLD``,
the birth-place toponym gate passes, and the passport is valid on the reference
date.  The product is <= the minimum field probability, so this is strictly
stricter than a per-field floor; it additionally penalises documents with
several moderately uncertain fields.  The toponym gate rejects a birth place
whose words are not all attested as gold-label tokens, which removes the
one-letter rare-toponym errors that the confidence model cannot see.
"""
from __future__ import annotations

import collections
import json
import math
import os
import zlib

import numpy as np
import pandas as pd

from pipeline import (FIELDS, build_resources, feature_dict, make_context,
                      select_fields)
from normalize import norm
from scoring import DocScorer, FieldModel
from validity import is_valid_on, parse_date

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
N_FOLDS = 5
DOC_THRESHOLD = 0.98
# Per-field floors are below DOC_THRESHOLD (the product implies every field is
# at least DOC_THRESHOLD); they are kept as an explicit safety net.
FIELD_FLOORS = {
    "Last_name_ru": 0.5,
    "First_name_ru": 0.5,
    "Middle_name_ru": 0.5,
    "Sex_ru": 0.9,
    "Birth_date": 0.9,
    "Birth_place_ru": 0.5,
    "Issue_date": 0.9,
    "Issue_organisation_code": 0.9,
    "Licence_number": 0.9,
    "Issue_organization_ru": 0.5,
}


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def fold_of(doc_id: str) -> int:
    return zlib.crc32(doc_id.encode("utf-8")) % N_FOLDS


def load_reference() -> tuple[set, collections.Counter]:
    frame = pd.read_csv(os.path.join(REFERENCE, "subdivisions.csv"),
                        sep=None, engine="python")
    codes = set(frame["code"].astype(str))
    tokens = collections.Counter()
    for name in frame["name"].astype(str):
        for token in norm(name).replace(",", " ").split():
            tokens[token] += 1
    return codes, tokens


def field_correct(candidate, gold: dict, field: str) -> bool:
    return candidate is not None and norm(candidate.value) == norm(gold.get(field))


def validity_of(candidates: dict) -> bool:
    birth = parse_date(candidates["Birth_date"].value)
    issue = parse_date(candidates["Issue_date"].value)
    return is_valid_on(birth, issue)


def raw_counts(docs) -> dict:
    counts = {name: collections.Counter() for name in FIELDS}
    for doc in docs:
        ctx = make_context(doc)
        for name in FIELDS:
            value = ctx.viz_text[name]
            if value:
                counts[name][value] += 1
    return counts


def build_oof(docs, labels, ref_codes, ref_tokens, batch_counts):
    rows = {name: [] for name in FIELDS}
    targets = {name: [] for name in FIELDS}
    meta = []
    for fold in range(N_FOLDS):
        train_docs = [doc for doc in docs if fold_of(doc["doc_id"]) != fold]
        resources = build_resources(train_docs, labels, ref_codes, ref_tokens)
        for doc in docs:
            if fold_of(doc["doc_id"]) != fold:
                continue
            candidates = select_fields(doc, resources, batch_counts)
            gold = labels[doc["doc_id"]]
            correct = {}
            for name in FIELDS:
                rows[name].append(feature_dict(candidates[name]))
                targets[name].append(int(field_correct(candidates[name], gold, name)))
                correct[name] = bool(targets[name][-1])
            meta.append({
                "doc_id": doc["doc_id"],
                "fold": fold,
                "source": doc.get("source"),
                "correct": correct,
                "all_correct": all(correct.values()),
                "valid": validity_of(candidates),
                "gate": bool(candidates["Birth_place_ru"].gate
                            and candidates["Last_name_ru"].gate),
                "toponym_gate": bool(candidates["Birth_place_ru"].gate),
                "surname_gate": bool(candidates["Last_name_ru"].gate),
                "valid_gold": is_valid_on(parse_date(gold.get("Birth_date")),
                                          parse_date(gold.get("Issue_date"))),
                "fields": {name: candidates[name].value for name in FIELDS},
                "reasons": {name: candidates[name].reason for name in FIELDS},
            })
    return rows, targets, meta


def rule_mask(prob: np.ndarray, doc_threshold: float,
              floors: np.ndarray) -> np.ndarray:
    return (prob >= floors).all(axis=1) & (prob.prod(axis=1) >= doc_threshold)


def main() -> None:
    os.makedirs(MODELS, exist_ok=True)
    docs = load_jsonl(os.path.join(DATA, "train_predictions.jsonl"))
    labels = {item["doc_id"]: item["fields"]
              for item in load_jsonl(os.path.join(DATA, "train_labels.jsonl"))}
    ref_codes, ref_tokens = load_reference()

    print("building out-of-fold features ...")
    counts = raw_counts(docs)
    rows, targets, meta = build_oof(docs, labels, ref_codes, ref_tokens, counts)
    n_docs = len(meta)
    folds = np.array([item["fold"] for item in meta])

    print("nested out-of-fold predictions ...")
    oof_prob = {name: np.zeros(n_docs) for name in FIELDS}
    for fold in range(N_FOLDS):
        train_idx = [i for i in range(n_docs) if folds[i] != fold]
        test_idx = [i for i in range(n_docs) if folds[i] == fold]
        for name in FIELDS:
            model = FieldModel.fit([rows[name][i] for i in train_idx],
                                   [targets[name][i] for i in train_idx])
            for i in test_idx:
                oof_prob[name][i] = model.proba(rows[name][i])

    prob = np.column_stack([oof_prob[name] for name in FIELDS])
    correct = np.array([[item["correct"][name] for name in FIELDS]
                        for item in meta], dtype=bool)
    valid = np.array([item["valid"] for item in meta])
    valid_gold = np.array([item["valid_gold"] for item in meta])
    all_correct = np.array([item["all_correct"] for item in meta])
    gate = np.array([item["gate"] for item in meta])
    good = valid & all_correct
    bad = valid & ~all_correct
    doc_prob = prob.prod(axis=1)
    floors = np.array([FIELD_FLOORS[name] for name in FIELDS])

    print("\nOOF per-field quality (fold models, threshold 0.5):")
    for k, name in enumerate(FIELDS):
        selected = prob[:, k] >= 0.5
        tp = int(np.sum(selected & correct[:, k]))
        print(f"  {name:28s} covered={int(selected.sum()):5d} "
              f"prec={tp / max(1, int(selected.sum())):.4f} "
              f"recall={tp / max(1, int(correct[:, k].sum())):.4f}")

    # ---- calibrated document rule on OOF ---------------------------------
    print("\ndocument-rule sensitivity on OOF (toponym + surname gates on):")
    for threshold in (0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98,
                      0.9825, 0.985, 0.9875, 0.99):
        accepted = rule_mask(prob, threshold, floors) & valid & gate
        n_good = int(np.sum(accepted & good))
        n_bad = int(np.sum(accepted & bad))
        print(f"  t={threshold:<7} accepted={n_good + n_bad:4d} "
              f"good={n_good:4d} bad={n_bad:2d} "
              f"err={n_bad / max(1, n_good + n_bad):.4f}")

    # Selected threshold.  Also evaluate the *procedure* in a nested way:
    # choose the best threshold on four folds and apply it to the fifth.
    accepted = rule_mask(prob, DOC_THRESHOLD, floors) & valid & gate
    n_good = int(np.sum(accepted & good))
    n_bad = int(np.sum(accepted & bad))
    print(f"\nselected t={DOC_THRESHOLD}: accepted={n_good + n_bad} "
          f"good={n_good} bad={n_bad}")

    nested_good = nested_bad = 0
    for fold in range(N_FOLDS):
        fit = folds != fold
        evaluate = folds == fold
        best = None
        for threshold in (0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.975, 0.98,
                          0.9825, 0.985, 0.9875, 0.99, 0.995):
            mask = rule_mask(prob[fit], threshold, floors) & valid[fit] & gate[fit]
            g = int(np.sum(mask & good[fit]))
            b = int(np.sum(mask & bad[fit]))
            if g and b == 0 and (best is None or g > best[0]):
                best = (g, threshold)
        threshold = best[1] if best else DOC_THRESHOLD
        mask = rule_mask(prob[evaluate], threshold, floors) & valid[evaluate] & gate[evaluate]
        g = int(np.sum(mask & good[evaluate]))
        b = int(np.sum(mask & bad[evaluate]))
        nested_good += g
        nested_bad += b
        print(f"  nested fold {fold}: t={threshold} good={g} bad={b}")
    print(f"nested procedure: good={nested_good} bad={nested_bad}")

    # ---- distributions for the report ------------------------------------
    print("\nvalidity: valid=%d invalid=%d; all-correct-but-invalid=%d"
          % (int(valid.sum()), int((~valid).sum()),
             int((all_correct & ~valid_gold).sum())))
    toponym_gate = np.array([item["toponym_gate"] for item in meta])
    surname_gate = np.array([item["surname_gate"] for item in meta])
    print("toponym gate: passed=%d failed=%d; all-correct passed=%d"
          % (int(toponym_gate.sum()), int((~toponym_gate).sum()),
             int((good & toponym_gate).sum())))
    print("surname gate: passed=%d failed=%d; all-correct passed=%d"
          % (int(surname_gate.sum()), int((~surname_gate).sum()),
             int((good & surname_gate).sum())))
    print("source mix (all / accepted):")
    for source in sorted({item["source"] for item in meta}):
        mask = np.array([item["source"] == source for item in meta])
        print(f"  {source:10s} {int(mask.sum()):4d} / "
              f"{int(np.sum(accepted & mask)):4d}")

    # 95% Wilson interval for the accepted-error rate.
    z = 1.96
    n = n_good + n_bad
    p = n_bad / max(1, n)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    print(f"error rate among accepted: {p:.4f} "
          f"(95% CI {max(0.0, centre - half):.4f}..{centre + half:.4f})")

    # ---- final model -----------------------------------------------------
    # Fit on the OOF feature rows, not on features built with the full label
    # set: the latter would contain each document's own label and make
    # ``val_train``/``name_unseen`` behave differently from inference time.
    print("\nfitting final models on the out-of-fold feature rows ...")
    final = {name: FieldModel.fit(rows[name], targets[name]) for name in FIELDS}
    full_resources = build_resources(docs, labels, ref_codes, ref_tokens)
    scorer = DocScorer(final, FIELD_FLOORS, DOC_THRESHOLD)
    with open(os.path.join(MODELS, "resources.json"), "w", encoding="utf-8") as handle:
        json.dump(full_resources.as_dict(), handle, ensure_ascii=False)
    print("writing models/scorer.json")
    scorer.save(os.path.join(MODELS, "scorer.json"), {
        "oof": {
            "n_docs": n_docs,
            "doc_threshold": DOC_THRESHOLD,
            "accepted": n_good + n_bad,
            "accepted_good": n_good,
            "accepted_bad": n_bad,
            "nested_good": nested_good,
            "nested_bad": nested_bad,
            "valid": int(valid.sum()),
            "all_correct": int(all_correct.sum()),
            "usable": int(good.sum()),
            "gate_passed": int(gate.sum()),
            "gate_passed_good": int((good & gate).sum()),
        }
    })

    # Save OOF predictions (with values) for reporting and error analysis.
    with open(os.path.join(MODELS, "oof.jsonl"), "w", encoding="utf-8") as handle:
        for i, item in enumerate(meta):
            record = {
                "doc_id": item["doc_id"], "fold": item["fold"],
                "source": item["source"], "all_correct": item["all_correct"],
                "valid": item["valid"], "valid_gold": item["valid_gold"],
                "doc_prob": float(doc_prob[i]),
                "prob": {name: float(oof_prob[name][i]) for name in FIELDS},
                "correct": item["correct"], "reasons": item["reasons"],
                "fields": item["fields"], "gate": item["gate"],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
