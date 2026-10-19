import pytest

from app import data
from app.scorer import ScoringInputError, load_gold, match_claims, score
from app.segment import segment

GOLD = load_gold()
TEXT = {a["answer_id"]: a["ai_answer"] for a in data.load_answers()}
DEV = [a["answer_id"] for a in data.load_answers() if a["split"] == "development"]
CHECK = [a["answer_id"] for a in data.load_answers() if a["split"] == "check"]


def gold_for(*ids):
    return [g for g in GOLD if g["answer_id"] in ids]


def pred(aid, s, e, label):
    return {"claim_id": f"{aid}:{s}:{e}", "start_char": s, "end_char": e, "quote": TEXT[aid][s:e],
            "label": label, "fact_ids": [], "reason": "Test."}


def from_gold(g, label=None):
    return pred(g["answer_id"], g["start_char"], g["end_char"], label or g["expected_label"])


def non_gold_segments(aid):
    spans = {(g["start_char"], g["end_char"]) for g in gold_for(aid)}
    return [(s, e) for _, s, e, _ in segment(TEXT[aid]) if (s, e) not in spans]


def result(aid, claims, status=None, error=None):
    status = status or ("error" if error else "ok" if claims else "empty")
    return {"answer_id": aid, "status": status, "claims": sorted(claims, key=lambda c: c["start_char"]),
            "error": error}


def perfect(ids):
    return [result(a, [from_gold(g) for g in gold_for(a)]) for a in ids]


def test_perfect_output_full_dev():
    out = score("development", DEV, perfect(DEV))
    assert (out["G"], out["P"], out["M"], out["C"]) == (376, 376, 376, 376)
    assert out["extraction"] == out["joint"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert out["coverage"] == 1.0 and out["missed_gold_ids"] == [] and out["wrong_label_ids"] == []
    assert out["counts"] == {"scheduled": 40, "ok": 40, "empty": 0, "error": 0}


def test_missed_claims():
    g = gold_for("A01")
    out = score("development", ["A01"], [result("A01", [from_gold(x) for x in g[:5]])])
    assert (out["G"], out["P"], out["M"]) == (9, 5, 5)
    assert out["missed_gold_ids"] == [x["claim_id"] for x in g[5:]]
    assert out["extraction"]["recall"] == pytest.approx(5 / 9)
    assert out["extraction"]["precision"] == 1.0


def test_extra_advice_sentence_extracted():
    s, e = non_gold_segments("A01")[0]  # sentence 0 is buying advice, not a gold claim
    claims = [from_gold(x) for x in gold_for("A01")] + [pred("A01", s, e, "unknown")]
    out = score("development", ["A01"], [result("A01", claims)])
    assert (out["P"], out["M"]) == (10, 9)
    assert out["unmatched_prediction_ids"] == [f"A01:{s}:{e}"]
    assert out["extraction"]["precision"] == pytest.approx(0.9)
    assert out["extraction"]["recall"] == 1.0


def test_wrong_labels():
    g = gold_for("A01")
    flip = {"supported": "unknown", "unknown": "contradicted", "contradicted": "supported"}
    claims = [from_gold(g[0], flip[g[0]["expected_label"]])] + [from_gold(x) for x in g[1:]]
    out = score("development", ["A01"], [result("A01", claims)])
    assert (out["M"], out["C"]) == (9, 8)
    assert out["wrong_label_ids"] == [{"claim_id": g[0]["claim_id"], "expected": g[0]["expected_label"],
                                       "predicted": flip[g[0]["expected_label"]]}]
    assert out["extraction"]["f1"] == 1.0 and out["joint"]["f1"] == pytest.approx(8 / 9)


def test_provider_error_answer_keeps_gold():
    err = result("A02", [], error={"type": "provider_error", "message": "x"})
    out = score("development", ["A01", "A02"], perfect(["A01"]) + [err])
    n2 = len(gold_for("A02"))
    assert out["G"] == 9 + n2 and out["M"] == 9
    assert set(x["claim_id"] for x in gold_for("A02")) <= set(out["missed_gold_ids"])
    assert out["error_counts"] == {"provider_error": 1}
    assert out["counts"]["error"] == 1


def test_valid_empty_predictions():
    out = score("development", ["A01"], [result("A01", [])])
    assert (out["G"], out["P"], out["M"], out["C"]) == (9, 0, 0, 0)
    assert out["extraction"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert out["counts"]["empty"] == 1


def _values(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _values(v)
    else:
        yield obj


def test_arithmetic_10_10_8_7():
    gold = gold_for("A01") + gold_for("A02")[:1]  # G = 10
    g1 = gold_for("A01")
    claims_a01 = [from_gold(x) for x in g1[:7]]  # 7 correct
    wrong_label = {"supported": "unknown", "unknown": "supported", "contradicted": "unknown"}
    claims_a01.append(from_gold(g1[7], wrong_label[g1[7]["expected_label"]]))  # 8th match, wrong label
    extras = [pred("A01", s, e, "unknown") for s, e in non_gold_segments("A01")[:2]]  # P = 10
    out = score("development", ["A01", "A02"],
                [result("A01", claims_a01 + extras), result("A02", [])], gold=gold)
    assert (out["G"], out["P"], out["M"], out["C"]) == (10, 10, 8, 7)
    assert out["extraction"] == pytest.approx({"precision": 0.8, "recall": 0.8, "f1": 0.8})
    assert out["joint"] == pytest.approx({"precision": 0.7, "recall": 0.7, "f1": 0.7})
    assert out["coverage"] == pytest.approx(0.7)
    # C/M = 0.875 is matched-only label accuracy and must not be reported
    assert not any(isinstance(v, float) and v == pytest.approx(0.875) for v in _values(out))
    assert not any(isinstance(k, str) and "accuracy" in k for k in _values(out))


def test_one_gold_cannot_match_two_predictions():
    g = gold_for("A01")[0]
    pairs, unmatched = match_claims([{**from_gold(g), "answer_id": "A01"}] * 2, [g])
    assert len(pairs) == 1 and len(unmatched) == 1


def test_check_split_nulls():
    preds = [pred("A41", s, e, "unknown") for _, s, e, _ in segment(TEXT["A41"])[:2]]
    out = score("check", CHECK, [result("A41", preds)] + [result(a, []) for a in CHECK[1:]])
    assert out["P"] == 2 and out["counts"]["scheduled"] == 10
    for k in ("G", "M", "C", "extraction", "joint", "coverage", "matched_gold_ids", "missed_gold_ids",
              "unmatched_prediction_ids", "wrong_label_ids", "per_tag"):
        assert out[k] is None, k


def test_empty_scheduled_split_nulls():
    out = score("development", [], [])
    assert out["extraction"] is None and out["joint"] is None and out["G"] is None


def test_missing_outcome_rejected():
    with pytest.raises(ScoringInputError, match="missing"):
        score("development", ["A01", "A02"], perfect(["A01"]))


def test_duplicated_outcome_rejected():
    with pytest.raises(ScoringInputError, match="duplicated"):
        score("development", ["A01"], perfect(["A01"]) * 2)


def test_unscheduled_outcome_rejected():
    with pytest.raises(ScoringInputError, match="unscheduled"):
        score("development", ["A01"], perfect(["A01", "A02"]))


def test_invalid_prediction_rejected():
    bad = {**from_gold(gold_for("A01")[0]), "quote": "not the text"}
    with pytest.raises(Exception, match="quote mismatch"):
        score("development", ["A01"], [result("A01", [bad])])


def test_per_tag_breakdown():
    out = score("development", ["A01"], [result("A01", [])])
    tags = {t for g in gold_for("A01") for t in g["difficulty_tags"]}
    assert set(out["per_tag"]) == tags
    assert all(v["missed"] == v["gold"] for v in out["per_tag"].values())
