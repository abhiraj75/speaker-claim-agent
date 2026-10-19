import json
import os

import psycopg
import pytest
from dotenv import load_dotenv
from psycopg import sql
from psycopg.conninfo import make_conninfo

from app import data

TEST_DB = "kasparro_test"


@pytest.fixture(scope="session")
def pg_url():
    """A separate test database next to DATABASE_URL, so tests never touch imported runs."""
    load_dotenv()
    base = os.getenv("DATABASE_URL")
    if not base:
        pytest.skip("DATABASE_URL not set: DB/API tests not run")
    try:
        with psycopg.connect(base, autocommit=True, connect_timeout=3) as c:
            if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
                c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(TEST_DB)))
    except psycopg.OperationalError:
        pytest.skip("DATABASE_URL unreachable: start Postgres (see README) to run DB/API tests")
    return make_conninfo(base, dbname=TEST_DB)


@pytest.fixture
def conn(pg_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    from app import db
    with db.connect(pg_url) as c:
        db.apply_schema(c)
        c.execute("TRUNCATE runs CASCADE")
        c.commit()
        yield c


def call(name, args, id="c1"):
    return {"id": id, "name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}


def scripted_split(split, error_index=2):
    """FakeProvider script for a whole split: answer 0 ok (one claim), error_index provider error, rest empty."""
    from app.providers import ProviderError
    script = []
    for i, a in enumerate(x for x in data.load_answers() if x["split"] == split):
        if i == error_index:
            script.append(ProviderError("APIStatusError 500: upstream"))
            continue
        script.append({"tool_calls": [call("get_brand_facts", {"brand_id": a["brand_id"]})]})
        claims = [{"sentence_index": 1, "label": "unknown", "fact_ids": [], "reason": "Not in the catalogue."}] if i == 0 else []
        script.append({"tool_calls": [call("submit_claims", {"claims": claims})]})
    return script
