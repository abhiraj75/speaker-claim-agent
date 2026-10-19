"""Postgres persistence. Parameterized SQL only."""
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

SCHEMA = Path(__file__).with_name("schema.sql")


def connect(url=None):
    load_dotenv()
    return psycopg.connect(url or os.environ["DATABASE_URL"], row_factory=dict_row, connect_timeout=5)


def apply_schema(conn):
    with conn.transaction():
        conn.execute(SCHEMA.read_text(encoding="utf-8"))


def run_exists(conn, run_id):
    return conn.execute("SELECT 1 FROM runs WHERE id = %s", (run_id,)).fetchone() is not None


def save_run(conn, report, dev_match=None):
    """Insert one run, one terminal row per answer, one row per accepted claim, in a single transaction."""
    dev_match = dev_match or {}
    model = report["resolved_model"] if isinstance(report["resolved_model"], str) else report["requested_model"]
    settings = {**report["settings"], "requested_model": report["requested_model"],
                "resolved_model": report["resolved_model"]}
    with conn.transaction():
        conn.execute(
            """INSERT INTO runs (id, split, agent_version, prompt_version, dataset_version, fact_version,
                   evaluator_version, provider, model, settings, file_hashes, source_revision, status, counts,
                   metrics, score_details, created_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            (report["run_id"], report["split"], report["agent_version"], report["prompt_version"],
             report["dataset_version"], report["fact_version"], report["evaluator_version"], report["provider"],
             model, Jsonb(settings), Jsonb(report["file_hashes"]), report["source_revision"], report["status"],
             Jsonb(report["counts"]), Jsonb(report["metrics"]), Jsonb(report["score_details"]),
             report["timestamp"]))
        for a in report["answers"]:
            conn.execute(
                """INSERT INTO answer_results (run_id, answer_id, status, error, latency_ms, trace_ref)
                   VALUES (%s, %s, %s, %s, %s, %s)""",
                (report["run_id"], a["answer_id"], a["status"], Jsonb(a["error"]) if a["error"] else None,
                 a["latency_ms"], a["trace_ref"]))
        for c in report["predictions"]:
            m = dev_match.get(c["claim_id"])
            conn.execute(
                """INSERT INTO claim_results (run_id, answer_id, claim_id, start_char, end_char, quote, label,
                       fact_ids, reason, dev_match)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (report["run_id"], c["answer_id"], c["claim_id"], c["start_char"], c["end_char"], c["quote"],
                 c["label"], c["fact_ids"], c["reason"], Jsonb(m) if m is not None else None))


def fetch_run(conn, run_id):
    """Run metadata + answers ordered by answer_id, each with claims ordered by start_char; None if missing."""
    run = conn.execute("SELECT * FROM runs WHERE id = %s", (run_id,)).fetchone()
    if run is None:
        return None
    answers = conn.execute(
        """SELECT answer_id, status, error, latency_ms, trace_ref FROM answer_results
           WHERE run_id = %s ORDER BY answer_id""", (run_id,)).fetchall()
    claims = conn.execute(
        """SELECT answer_id, claim_id, start_char, end_char, quote, label, fact_ids, reason, dev_match
           FROM claim_results WHERE run_id = %s ORDER BY answer_id, start_char""", (run_id,)).fetchall()
    by_answer = {a["answer_id"]: {**a, "claims": []} for a in answers}
    for c in claims:
        by_answer[c.pop("answer_id")]["claims"].append(c)
    return {"run": run, "answers": list(by_answer.values())}
