"""Thin FastAPI routes over the agent, runner and db."""
import uuid
from typing import Literal

import psycopg
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from app import db, runner
from app.agent import run_agent
from app.data import BRANDS

app = FastAPI(title="Kasparro claim agent")
RUNS_DIR = runner.RUNS_DIR


def get_provider():
    from app.providers import OpenAIProvider
    return OpenAIProvider()


def get_conn():
    with db.connect() as conn:
        yield conn


def error(status, type, message):
    return JSONResponse(status_code=status, content={"type": type, "message": message})


@app.exception_handler(RequestValidationError)
def _invalid_request(request: Request, exc: RequestValidationError):
    fields = sorted({".".join(str(p) for p in e["loc"][1:]) or e["loc"][0] for e in exc.errors()})
    return error(422, "invalid_request", f"invalid or unexpected fields: {', '.join(fields)}")


@app.exception_handler(psycopg.OperationalError)
def _db_down(request: Request, exc: psycopg.OperationalError):
    return error(503, "database_unavailable", "database is not reachable")


class ClassifyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_id: str = Field(min_length=1)
    brand_id: str = Field(min_length=1)
    question: str
    ai_answer: str = Field(min_length=1)
    agent_version: Literal["v1", "v2"]


class EvaluationRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    agent_version: Literal["v1", "v2"]
    split: Literal["development", "check"]


ERROR_STATUS = {"timeout": 504}  # every other agent error type is 502


@app.post("/classify")
def classify(req: ClassifyRequest, provider=Depends(get_provider)):
    if req.brand_id not in BRANDS:  # checked before any model call
        return error(404, "unknown_brand", f"unknown primary brand {req.brand_id[:40]!r}")
    inp = req.model_dump(exclude={"agent_version"})
    r = run_agent(inp, req.agent_version, provider)
    if r.status == "error":
        return error(ERROR_STATUS.get(r.error["type"], 502), r.error["type"], r.error["message"])
    return {"answer_id": r.answer_id, "claims": [c.model_dump() for c in r.claims]}


@app.post("/evaluation-runs", status_code=201)
def create_evaluation_run(req: EvaluationRunRequest, provider=Depends(get_provider), conn=Depends(get_conn)):
    report = runner.run_split(req.split, req.agent_version, provider, runs_dir=RUNS_DIR,
                              persist=lambda rep: runner.persist(conn, rep))
    return {k: report[k] for k in ("run_id", "status", "counts", "metrics")}


@app.get("/evaluation-runs/{run_id}")
def get_evaluation_run(run_id: uuid.UUID, conn=Depends(get_conn)):
    found = db.fetch_run(conn, run_id)
    if found is None:
        return error(404, "not_found", f"run {run_id} not found")
    return found
