import json

import pytest

from app import data
from app.providers import FakeProvider, ProviderError
from app.runner import run_split


def call(name, args, id="c1"):
    return {"id": id, "name": name, "arguments": json.dumps(args)}


def script_ok_empty_error(brands):
    return [
        {"tool_calls": [call("get_brand_facts", {"brand_id": brands[0]})]},
        {"tool_calls": [call("submit_claims", {"claims": [
            {"sentence_index": 1, "label": "unknown", "fact_ids": [], "reason": "Not in the catalogue."}]})]},
        {"tool_calls": [call("get_brand_facts", {"brand_id": brands[1]})]},
        {"tool_calls": [call("submit_claims", {"claims": []})]},
        ProviderError("APIStatusError 500: upstream"),
    ]


def ids_and_brands(split):
    rows = [a for a in data.load_answers() if a["split"] == split][:3]
    return [a["answer_id"] for a in rows], [a["brand_id"] for a in rows]


def test_development_run_report(tmp_path):
    ids, brands = ids_and_brands("development")
    persisted = []
    report = run_split("development", "v1", FakeProvider(script_ok_empty_error(brands)),
                       runs_dir=tmp_path, answer_ids=ids, persist=persisted.append)
    saved = json.loads((tmp_path / f"{report['run_id']}.json").read_text())
    assert saved == json.loads(json.dumps(report)) and persisted == [report]
    assert saved["status"] == "completed"
    assert saved["counts"] == {"scheduled": 3, "ok": 1, "empty": 1, "error": 1}
    assert [a["status"] for a in saved["answers"]] == ["ok", "empty", "error"]
    assert saved["answers"][2]["error"]["type"] == "provider_error"
    for a in saved["answers"]:
        assert (tmp_path.parent / a["trace_ref"]).exists()
    for k in ("run_id", "split", "agent_version", "prompt_version", "dataset_version", "fact_version",
              "evaluator_version", "provider", "requested_model", "resolved_model", "settings", "timestamp",
              "source_revision", "file_hashes", "timing", "predictions"):
        assert saved[k] is not None, k
    assert saved["file_hashes"] == data.EXPECTED_HASHES
    assert saved["settings"]["max_turns"] == 3 and saved["prompt_version"].startswith("v1-")
    assert len(saved["predictions"]) == 1 and saved["predictions"][0]["answer_id"] == ids[0]
    m = saved["metrics"]
    gold = [g for g in data.read_csv("development_gold.csv") if g["answer_id"] in ids]
    assert m["G"] == len(gold) and m["P"] == 1
    assert saved["score_details"]["error_counts"] == {"provider_error": 1}
    assert set(saved["score_details"]["records"]) == {"missed", "extra", "wrong_label"}


def test_check_run_metrics_null(tmp_path):
    ids, brands = ids_and_brands("check")
    report = run_split("check", "v1", FakeProvider(script_ok_empty_error(brands)), runs_dir=tmp_path, answer_ids=ids)
    assert report["status"] == "completed"
    assert report["counts"] == {"scheduled": 3, "ok": 1, "empty": 1, "error": 1}
    assert all(report["metrics"][k] is None for k in ("G", "M", "C", "extraction", "joint", "coverage"))
    assert report["metrics"]["P"] == 1
    assert report["score_details"]["missed_gold_ids"] is None


def test_interrupted_run_is_failed_with_partial_rows(tmp_path):
    ids, brands = ids_and_brands("development")
    script = script_ok_empty_error(brands)[:2] + [RuntimeError("disk full")]
    with pytest.raises(RuntimeError):
        run_split("development", "v1", FakeProvider(script), runs_dir=tmp_path, answer_ids=ids)
    [path] = tmp_path.glob("*.json")
    saved = json.loads(path.read_text())
    assert saved["status"] == "failed" and saved["failure"]["type"] == "RuntimeError"
    assert [a["answer_id"] for a in saved["answers"]] == ids[:1]
    assert saved["metrics"] is None
