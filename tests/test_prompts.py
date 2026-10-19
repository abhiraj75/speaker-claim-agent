from app.agent import PROMPTS_DIR, load_prompt
from app.runner import prompt_version

V1_FROZEN = "v1-5852cc74cc26"  # prompt_version recorded by the v1 development run


def test_v1_prompt_is_frozen():
    assert prompt_version("v1") == V1_FROZEN


def test_v2_is_v1_plus_exactly_one_rule():
    v1 = load_prompt("v1").splitlines()
    v2 = load_prompt("v2").splitlines()
    added = [line for line in v2 if line not in v1]
    assert len(v2) == len(v1) + 1 and len(added) == 1
    assert [line for line in v2 if line in v1] == v1  # order of v1 lines preserved
    assert "shipping" in added[0]


def test_prompts_carry_no_gold_field_names():
    for v in ("v1", "v2"):
        text = (PROMPTS_DIR / f"{v}.md").read_text(encoding="utf-8")
        for f in ("expected_label", "evidence_fact_ids", "difficulty_tags", "development_gold"):
            assert f not in text
