"""Deterministic scorer: exact span matching, extraction and joint P/R/F1. Gold is read only here and in tests."""
from collections import Counter

from app import data
from app.validate import validate_claims

GOLD_FILE = "development_gold.csv"


class ScoringInputError(Exception):
    pass


def load_gold():
    rows = data.read_csv(GOLD_FILE)
    return [{**g, "start_char": int(g["start_char"]), "end_char": int(g["end_char"]),
             "difficulty_tags": [t for t in g["difficulty_tags"].split("|") if t]} for g in rows]


def _prf(hits, p, g):
    return {"precision": hits / p if p else 0.0, "recall": hits / g if g else 0.0,
            "f1": 2 * hits / (p + g) if p + g else 0.0}


def match_claims(preds, gold):
    """Pair predictions with gold on identical (answer_id, start_char, end_char); each gold matches at most once."""
    by_span = {(g["answer_id"], g["start_char"], g["end_char"]): g for g in gold}
    used, pairs, unmatched = set(), [], []
    for p in preds:
        key = (p["answer_id"], p["start_char"], p["end_char"])
        if key in by_span and key not in used:
            used.add(key)
            pairs.append((p, by_span[key]))
        else:
            unmatched.append(p)
    return pairs, unmatched


def _outcomes(scheduled_ids, results):
    results = [r if isinstance(r, dict) else r.model_dump() for r in results]
    seen = Counter(r["answer_id"] for r in results)
    if dup := sorted(a for a, n in seen.items() if n > 1):
        raise ScoringInputError(f"duplicated answer outcomes: {dup}")
    if missing := sorted(set(scheduled_ids) - set(seen)):
        raise ScoringInputError(f"missing answer outcomes: {missing}")
    if extra := sorted(set(seen) - set(scheduled_ids)):
        raise ScoringInputError(f"outcomes for unscheduled answers: {extra}")
    return results


def score(split, scheduled_ids, results, gold=None):
    """Score one split. Check split: every gold-dependent field is None. gold defaults to the development file."""
    if len(set(scheduled_ids)) != len(scheduled_ids):
        raise ScoringInputError("scheduled answer_ids repeat")
    results = _outcomes(scheduled_ids, results)
    texts = {a["answer_id"]: a["ai_answer"] for a in data.load_answers()}
    preds = []
    for r in results:
        if r["status"] == "error" and r["claims"]:
            raise ScoringInputError(f"{r['answer_id']}: error outcome carries claims")
        for c in validate_claims(texts[r["answer_id"]], r["claims"], answer_id=r["answer_id"]):
            preds.append({"answer_id": r["answer_id"], **c.model_dump()})

    counts = {"scheduled": len(scheduled_ids), **{s: sum(r["status"] == s for r in results) for s in ("ok", "empty", "error")}}
    out = {"split": split, "counts": counts,
           "error_counts": dict(Counter(r["error"]["type"] for r in results if r["status"] == "error")),
           "G": None, "P": len(preds), "M": None, "C": None, "extraction": None, "joint": None, "coverage": None,
           "matched_gold_ids": None, "missed_gold_ids": None, "unmatched_prediction_ids": None,
           "wrong_label_ids": None, "per_tag": None}
    if split == "check" or not scheduled_ids:
        return out
    if split != "development":
        raise ScoringInputError(f"unknown split {split!r}")

    scheduled = set(scheduled_ids)
    gold = [g for g in (load_gold() if gold is None else gold) if g["answer_id"] in scheduled]
    pairs, unmatched = match_claims(preds, gold)
    matched = {g["claim_id"] for _, g in pairs}
    wrong = [{"claim_id": g["claim_id"], "expected": g["expected_label"], "predicted": p["label"]}
             for p, g in pairs if p["label"] != g["expected_label"]]
    G, P, M = len(gold), len(preds), len(pairs)
    C = M - len(wrong)
    wrong_ids = {w["claim_id"] for w in wrong}
    per_tag = {}
    for g in gold:
        for t in g["difficulty_tags"]:
            s = per_tag.setdefault(t, {"gold": 0, "correct": 0, "missed": 0, "wrong_label": 0})
            s["gold"] += 1
            k = "missed" if g["claim_id"] not in matched else "wrong_label" if g["claim_id"] in wrong_ids else "correct"
            s[k] += 1
    out.update({
        "G": G, "M": M, "C": C,
        "extraction": _prf(M, P, G) if G else None,
        "joint": _prf(C, P, G) if G else None,
        "coverage": C / G if G else None,
        "matched_gold_ids": sorted(matched),
        "missed_gold_ids": [g["claim_id"] for g in gold if g["claim_id"] not in matched],
        "unmatched_prediction_ids": [p["claim_id"] for p in unmatched],
        "wrong_label_ids": wrong,
        "per_tag": dict(sorted(per_tag.items(), key=lambda kv: -kv[1]["gold"])),
    })
    return out
