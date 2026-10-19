import uuid

import pytest
from fastapi.testclient import TestClient

from app import api, data
from app.providers import FakeProvider, ProviderTimeout
from tests.conftest import call, scripted_split

A01 = next(a for a in data.load_answers() if a["answer_id"] == "A01")
BODY = {k: A01[k] for k in ("answer_id", "brand_id", "question", "ai_answer")} | {"agent_version": "v1"}
LOOKUP = {"tool_calls": [call("get_brand_facts", {"brand_id": "cleartrail"})]}


def submit(claims):
    return {"tool_calls": [call("submit_claims", {"claims": claims}, "c2")]}


@pytest.fixture
def client():
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def use(script):
    p = FakeProvider(script)
    api.app.dependency_overrides[api.get_provider] = lambda: p
    return p


def assert_error_body(r, status, type):
    assert r.status_code == status, r.text
    body = r.json()
    assert set(body) == {"type", "message"} and body["type"] == type
    assert "Traceback" not in r.text and "sk-" not in r.text


def test_classify_200(client):
    use([LOOKUP, submit([{"sentence_index": 3, "label": "unknown", "fact_ids": [], "reason": "No fact."}])])
    r = client.post("/classify", json=BODY)
    assert r.status_code == 200
    [c] = r.json()["claims"]
    assert c["quote"] == A01["ai_answer"][c["start_char"]:c["end_char"]]
    assert c["claim_id"] == f"A01:{c['start_char']}:{c['end_char']}"


def test_classify_valid_empty_200(client):
    use([LOOKUP, submit([])])
    r = client.post("/classify", json=BODY)
    assert r.status_code == 200 and r.json() == {"answer_id": "A01", "claims": []}


@pytest.mark.parametrize("body", [
    {k: v for k, v in BODY.items() if k != "ai_answer"},
    BODY | {"claims": []},
    BODY | {"expected_label": "supported"},
    BODY | {"agent_version": "v3"},
    BODY | {"ai_answer": ""},
], ids=["missing_field", "extra_claims", "extra_gold_field", "bad_agent_version", "empty_answer"])
def test_classify_422(client, body):
    p = use([])
    r = client.post("/classify", json=body)
    assert_error_body(r, 422, "invalid_request")
    assert p.calls == []


def test_classify_404_unknown_brand_no_model_call(client):
    p = use([LOOKUP])
    r = client.post("/classify", json=BODY | {"brand_id": "acme"})
    assert_error_body(r, 404, "unknown_brand")
    assert p.calls == []


@pytest.mark.parametrize("script,status,type", [
    ([LOOKUP, submit([{"sentence_index": 0, "label": "maybe", "fact_ids": [], "reason": "x"}])], 502, "invalid_output"),
    ([{"tool_calls": [call("delete_records", "{}")]}], 502, "illegal_tool"),
    ([{"tool_calls": [call("get_brand_facts", {"brand_id": "acme"})]}], 502, "unknown_brand"),
    ([LOOKUP, LOOKUP, LOOKUP], 502, "turn_limit"),
    ([ProviderTimeout("provider request exceeded 60s")], 504, "timeout"),
], ids=["invalid_output", "illegal_tool", "model_unknown_brand", "turn_limit", "timeout"])
def test_classify_errors(client, script, status, type):
    use(script)
    assert_error_body(client.post("/classify", json=BODY), status, type)


def test_classify_provider_error_502(client):
    from app.providers import ProviderError
    use([ProviderError("APIStatusError 500: upstream")])
    assert_error_body(client.post("/classify", json=BODY), 502, "provider_error")


def test_evaluation_run_and_get(client, conn, tmp_path, monkeypatch):
    monkeypatch.setattr(api, "RUNS_DIR", tmp_path)
    use(scripted_split("check"))
    r = client.post("/evaluation-runs", json={"agent_version": "v1", "split": "check"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["status"] == "completed"
    assert body["counts"] == {"scheduled": 10, "ok": 1, "empty": 8, "error": 1}
    assert all(body["metrics"][k] is None for k in ("G", "M", "C", "extraction", "joint", "coverage"))

    got = client.get(f"/evaluation-runs/{body['run_id']}")
    assert got.status_code == 200
    j = got.json()
    ids = [a["answer_id"] for a in j["answers"]]
    assert ids == sorted(ids) and len(ids) == 10
    assert {a["status"] for a in j["answers"]} == {"ok", "empty", "error"}
    err = next(a for a in j["answers"] if a["status"] == "error")
    assert err["error"]["type"] == "provider_error" and err["claims"] == []
    ok = next(a for a in j["answers"] if a["status"] == "ok")
    assert [c["start_char"] for c in ok["claims"]] == sorted(c["start_char"] for c in ok["claims"])
    assert j["run"]["split"] == "check" and j["run"]["metrics"]["G"] is None
    assert got.json() == client.get(f"/evaluation-runs/{body['run_id']}").json()  # stable


@pytest.mark.parametrize("body", [{"agent_version": "v1"}, {"agent_version": "v1", "split": "x"},
                                  {"agent_version": "v1", "split": "check", "gold": []}])
def test_evaluation_run_422(client, conn, body):
    use([])
    assert_error_body(client.post("/evaluation-runs", json=body), 422, "invalid_request")


def test_get_404_and_422(client, conn):
    assert_error_body(client.get(f"/evaluation-runs/{uuid.uuid4()}"), 404, "not_found")
    assert_error_body(client.get("/evaluation-runs/not-a-uuid"), 422, "invalid_request")
