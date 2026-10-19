import psycopg
import pytest

from app import db
from app.providers import FakeProvider
from app.runner import import_report, persist, run_split
from tests.conftest import scripted_split


@pytest.fixture
def dev_report(tmp_path):
    ids = ["A01", "A02", "A03"]
    return run_split("development", "v1", FakeProvider(scripted_split("development")), runs_dir=tmp_path,
                     answer_ids=ids)


def test_round_trip_ok_empty_error(conn, dev_report):
    persist(conn, dev_report)
    got = db.fetch_run(conn, dev_report["run_id"])
    assert str(got["run"]["id"]) == dev_report["run_id"] and got["run"]["status"] == "completed"
    assert got["run"]["file_hashes"] == dev_report["file_hashes"]
    assert [a["answer_id"] for a in got["answers"]] == ["A01", "A02", "A03"]
    assert [a["status"] for a in got["answers"]] == ["ok", "empty", "error"]
    assert got["answers"][2]["error"]["type"] == "provider_error" and got["answers"][2]["claims"] == []
    [claim] = got["answers"][0]["claims"]
    assert claim["claim_id"] == dev_report["predictions"][0]["claim_id"]
    assert claim["dev_match"] is not None


def test_one_terminal_row_per_scheduled_answer(conn, dev_report):
    persist(conn, dev_report)
    n = conn.execute("SELECT count(*) AS n FROM answer_results WHERE run_id = %s", (dev_report["run_id"],)).fetchone()
    assert n["n"] == dev_report["counts"]["scheduled"] == 3


def test_duplicate_answer_row_rejected(conn, dev_report):
    persist(conn, dev_report)
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute("INSERT INTO answer_results (run_id, answer_id, status) VALUES (%s, %s, 'empty')",
                     (dev_report["run_id"], "A02"))
    conn.rollback()


def test_duplicate_claim_span_rejected(conn, dev_report):
    persist(conn, dev_report)
    c = dev_report["predictions"][0]
    with pytest.raises(psycopg.errors.UniqueViolation):
        conn.execute(
            """INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char, quote, label,
                   fact_ids, reason) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (dev_report["run_id"], c["answer_id"], c["claim_id"], c["start_char"], c["end_char"], c["quote"],
             "supported", [], "dup"))
    conn.rollback()


def test_claim_requires_answer_row(conn, dev_report):
    persist(conn, dev_report)
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        conn.execute(
            """INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char, quote, label,
                   fact_ids, reason) VALUES (%s, 'A99', 'A99:0:1', 0, 1, 'x', 'unknown', '{}', 'r')""",
            (dev_report["run_id"],))
    conn.rollback()


def test_import_is_idempotent_and_recomputes_dev_match(conn, dev_report, tmp_path):
    path = tmp_path / f"{dev_report['run_id']}.json"
    assert import_report(path, conn) is True
    assert import_report(path, conn) is False
    got = db.fetch_run(conn, dev_report["run_id"])
    assert all(c["dev_match"] is not None for a in got["answers"] for c in a["claims"])


def test_check_run_has_no_dev_match(conn, tmp_path):
    rep = run_split("check", "v1", FakeProvider(scripted_split("check")), runs_dir=tmp_path)
    persist(conn, rep)
    got = db.fetch_run(conn, rep["run_id"])
    assert len(got["answers"]) == 10
    assert all(c["dev_match"] is None for a in got["answers"] for c in a["claims"])
    assert got["run"]["metrics"]["G"] is None
