import pytest

from app import data


def test_file_hashes_match_expected():
    assert data.file_hashes() == data.EXPECTED_HASHES


def test_counts():
    answers = data.load_answers()
    assert len(answers) == 50
    assert sum(a["split"] == "development" for a in answers) == 40
    assert sum(a["split"] == "check" for a in answers) == 10
    assert [a["answer_id"] for a in answers] == [f"A{i:02d}" for i in range(1, 51)]
    assert len(data.load_facts()) == 132
    assert len(data.read_csv("development_gold.csv")) == 376


def test_quoted_commas_and_embedded_newlines_parse():
    answers = data.load_answers()
    assert all("\n" in a["ai_answer"] for a in answers)
    assert any("," in a["ai_answer"] for a in answers)
    assert {a["brand_id"] for a in answers} == set(data.BRANDS)


def test_booleans_are_real_bools():
    bools = [f for f in data.load_facts() if f["value_type"] == "boolean"]
    assert bools and all(type(f["value"]) is bool for f in bools)
    assert any(f["value"] is False for f in bools)
    assert data.parse_value("false", "boolean") is False
    assert data.parse_value("true", "boolean") is True
    with pytest.raises(ValueError):
        data.parse_value("yes", "boolean")


def test_numbers_and_blanks():
    assert data.parse_value("1299", "number") == 1299
    assert data.parse_value("2.5", "number") == 2.5
    facts = data.load_facts()
    assert all(f["unit"] is None or f["unit"] for f in facts)
    assert any(f["condition"] is None for f in facts)


def test_six_models_three_brands():
    models = {(f["brand_id"], f["model_id"]) for f in data.load_facts()}
    assert len(models) == 6 and {b for b, _ in models} == set(data.BRANDS)


def test_fact_index():
    assert len(data.fact_ids()) == 132
    assert sum(len(v) for v in data.facts_by_brand().values()) == 132
