import pytest

from app import data
from app.segment import segment
from app.validate import OutputValidationError, build_claims, validate_claims

A = next(a for a in data.load_answers() if a["answer_id"] == "A01")
TEXT = A["ai_answer"]
SEGS = segment(TEXT)
FACT = sorted(data.fact_ids())[0]


def sel(i, **kw):
    return {"sentence_index": i, "label": "supported", "fact_ids": [FACT], "reason": "One. Two.", **kw}


def valid_claims():
    return [c.model_dump() for c in build_claims("A01", TEXT, SEGS, [sel(3), sel(1)])]


def test_build_claims_valid_offsets():
    claims = build_claims("A01", TEXT, SEGS, [sel(3), sel(1, label="unknown", fact_ids=[])])
    assert [c.start_char for c in claims] == [SEGS[1][1], SEGS[3][1]]
    for c in claims:
        assert c.quote == TEXT[c.start_char:c.end_char]
        assert c.claim_id == f"A01:{c.start_char}:{c.end_char}"


def test_offsets_are_code_points():
    text = "Prix €99 — ok. Next one."  # € and — are 3 bytes each in UTF-8
    [c] = build_claims("X", text, segment(text), [sel(1)])
    assert (c.start_char, c.end_char, c.quote) == (15, 24, "Next one.")


def test_valid_empty_list():
    assert build_claims("A01", TEXT, SEGS, []) == []
    assert validate_claims(TEXT, []) == []


def test_validate_claims_accepts_valid():
    assert len(validate_claims(TEXT, valid_claims(), answer_id="A01")) == 2


def _with(i, **kw):
    claims = valid_claims()
    c = claims[i]
    c.update(kw)
    if "quote" not in kw and ("start_char" in kw or "end_char" in kw):
        c["quote"] = TEXT[c["start_char"]:c["end_char"]]
    if "claim_id" not in kw:
        c["claim_id"] = f"A01:{c['start_char']}:{c['end_char']}"
    return claims


def _bad_cases():
    c0, c1 = valid_claims()
    n = len(TEXT)
    return {
        "end_out_of_range": (_with(1, end_char=n + 5), "out of range"),
        "negative_start": (_with(0, start_char=-1), "out of range"),
        "start_ge_end": (_with(0, end_char=c0["start_char"]), "out of range"),
        "quote_mismatch": (_with(0, quote=c0["quote"] + "x"), "quote mismatch"),
        "leading_whitespace": (_with(0, start_char=c0["start_char"] - 1), "whitespace"),
        "bad_claim_id": (_with(0, claim_id="A02:1:2"), "bad claim_id"),
        "duplicate_ids": ([c0, dict(c0)], "duplicate"),
        "unsorted": ([c1, c0], "unsorted"),
        "overlapping": (_with(1, start_char=c0["end_char"] - 3), "overlapping"),
        "bad_label": (_with(0, label="partially_supported"), "invalid Claim"),
        "invented_fact_id": (_with(0, fact_ids=["cleartrail.mini.made_up"]), "unknown fact_ids"),
        "reason_over_2_sentences": (_with(0, reason="One. Two. Three."), "over 2 sentences"),
        "extra_field": (_with(0, gold_label="supported"), "invalid Claim"),
    }


@pytest.mark.parametrize("case", list(_bad_cases()))
def test_validate_claims_rejects(case):
    claims, msg = _bad_cases()[case]
    with pytest.raises(OutputValidationError, match=msg):
        validate_claims(TEXT, claims, answer_id="A01")


@pytest.mark.parametrize("selections,msg", [
    ([sel(len(SEGS))], "out of range"),
    ([sel(-1)], "out of range"),
    ([sel(2), sel(2)], "repeated"),
    ([sel(0, label="maybe")], "invalid Selection"),
    ([sel(0, fact_ids=["nope"])], "unknown fact_ids"),
    ([sel(0, reason="A. B. C.")], "over 2 sentences"),
    ([sel("0")], "invalid Selection"),
    ([sel(0, start_char=0)], "invalid Selection"),
    ({"sentence_index": 0}, "must be a list"),
])
def test_build_claims_rejects(selections, msg):
    with pytest.raises(OutputValidationError, match=msg):
        build_claims("A01", TEXT, SEGS, selections)
