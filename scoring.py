"""Per-field correctness models and document-level accept rule.

The model is a logistic regression over the feature dictionary produced by
``pipeline.feature_dict``.  We store it as JSON (no pickle) so that the exact
weights used to produce ``answers.jsonl`` live next to the code.

Document rule
-------------
Each field gets an independent probability that the selected candidate equals
the reference label.  A document is accepted when

* every field has a value,
* every field clears its own (low) safety floor,
* the birth-place toponym gate passes (all words are attested, the value is
  complete and contains a concrete toponym),
* the surname gate passes (no unresolved Ш/Щ ambiguity),
* the product of the ten probabilities is at least ``doc_threshold``.

The product is a simple, calibrated joint-confidence score: it is at most the
smallest field probability, and it penalises documents in which several fields
are only moderately certain.  The threshold was selected on out-of-fold
predictions (see ``train.py`` and ``REPORT.md``).
"""
from __future__ import annotations

import json
import math
from typing import Optional

from pipeline import FIELDS


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class FieldModel:
    """Logistic regression with a hand-rolled sparse dict vectoriser."""

    def __init__(self, vocabulary: Optional[dict] = None, scale: Optional[list] = None,
                 coef: Optional[list] = None, intercept: float = 0.0):
        self.vocabulary = vocabulary or {}
        self.scale = scale or []
        self.coef = coef or []
        self.intercept = intercept

    @classmethod
    def fit(cls, rows: list[dict], target: list[int], *, c: float = 0.5):
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        vectorizer = DictVectorizer(sort=True)
        matrix = vectorizer.fit_transform(rows)
        scaler = StandardScaler(with_mean=False)
        scaled = scaler.fit_transform(matrix)
        model = LogisticRegression(C=c, max_iter=5000, solver="lbfgs")
        model.fit(scaled, target)
        names = vectorizer.get_feature_names_out()
        return cls(
            vocabulary={str(name): i for i, name in enumerate(names)},
            scale=[float(x) if x else 1.0 for x in scaler.scale_],
            coef=[float(x) for x in model.coef_[0]],
            intercept=float(model.intercept_[0]),
        )

    def logit(self, features: dict) -> float:
        total = self.intercept
        for key, value in features.items():
            index = self.vocabulary.get(key)
            if index is None:
                continue
            total += self.coef[index] * (float(value) / self.scale[index])
        return total

    def proba(self, features: dict) -> float:
        return _sigmoid(self.logit(features))

    def to_dict(self) -> dict:
        return {
            "vocabulary": self.vocabulary,
            "scale": self.scale,
            "coef": self.coef,
            "intercept": self.intercept,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FieldModel":
        return cls(data["vocabulary"], data["scale"], data["coef"], data["intercept"])


class DocScorer:
    """Accept a document when all field candidates are jointly confident."""

    def __init__(self, models: dict, thresholds: Optional[dict] = None,
                 doc_threshold: float = 0.0, toponym_gate: bool = True,
                 name_gate: bool = True):
        self.models = models
        self.thresholds = thresholds or {}
        self.doc_threshold = float(doc_threshold)
        self.toponym_gate = bool(toponym_gate)
        self.name_gate = bool(name_gate)

    @classmethod
    def load(cls, path: str) -> "DocScorer":
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        models = {name: FieldModel.from_dict(value)
                  for name, value in data["models"].items()}
        rule = data["rule"]
        thresholds = rule.get("thresholds") or {}
        if not thresholds and "min_prob" in rule:  # backwards compatibility
            thresholds = {name: float(rule["min_prob"]) for name in FIELDS}
        return cls(models,
                   {name: float(value) for name, value in thresholds.items()},
                   float(rule.get("doc_threshold", 0.0)),
                   bool(rule.get("toponym_gate", True)),
                   bool(rule.get("name_gate", True)))

    def save(self, path: str, extra: Optional[dict] = None) -> None:
        data = {
            "models": {name: model.to_dict() for name, model in self.models.items()},
            "rule": {
                "thresholds": {name: float(value)
                               for name, value in self.thresholds.items()},
                "doc_threshold": float(self.doc_threshold),
                "toponym_gate": self.toponym_gate,
                "name_gate": self.name_gate,
            },
        }
        if extra:
            data.update(extra)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=1)

    def score(self, candidates: dict) -> tuple[dict, dict, float, bool]:
        """Return (probabilities, logits, document probability, accepted)."""
        from pipeline import feature_dict
        probabilities, logits = {}, {}
        accepted = True
        for name, candidate in candidates.items():
            features = feature_dict(candidate)
            probabilities[name] = self.models[name].proba(features)
            logits[name] = self.models[name].logit(features)
            if candidate.value is None:
                accepted = False
            if probabilities[name] < self.thresholds.get(name, 0.0):
                accepted = False
        if self.toponym_gate and not candidates["Birth_place_ru"].gate:
            accepted = False
        if self.name_gate and not candidates["Last_name_ru"].gate:
            accepted = False
        document_probability = math.prod(probabilities.values())
        if document_probability < self.doc_threshold:
            accepted = False
        return probabilities, logits, document_probability, accepted
