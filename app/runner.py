"""Sequential evaluation runner: predictions are saved before gold is joined.

python -m app.runner --version v1 --split development
"""
import argparse
import hashlib
import json
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from app import data
from app.agent import MAX_TURNS, PROMPTS_DIR, load_prompt, run_agent
from app.providers import sanitize

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
SPLITS = ("development", "check")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def prompt_version(agent_version):
    digest = hashlib.sha256((PROMPTS_DIR / f"{agent_version}.md").read_bytes()).hexdigest()[:12]
    return f"{agent_version}-{digest}"


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_split(split, agent_version, provider, runs_dir=RUNS_DIR, answer_ids=None, persist=None, log=None):
    """Run the agent over a split, write runs/<run_id>.json, score development after saving predictions."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    load_prompt(agent_version)  # fail fast on a bad version or empty prompt
    answers = [a for a in data.load_answers() if a["split"] == split
               and (answer_ids is None or a["answer_id"] in answer_ids)]
    run_id = str(uuid.uuid4())
    report = {
        "run_id": run_id, "split": split, "agent_version": agent_version,
        "prompt_version": prompt_version(agent_version),
        "dataset_version": data.DATASET_VERSION, "fact_version": data.FACT_VERSION,
        "evaluator_version": data.EVALUATOR_VERSION,
        "provider": provider.name, "requested_model": provider.model, "resolved_model": None,
        "settings": {**provider.settings, "timeout_s": provider.timeout, "max_turns": MAX_TURNS},
        "timestamp": _now(), "source_revision": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain", "--", "app", "prompts")),
        "file_hashes": data.file_hashes(),
        "status": "running", "failure": None,
        "counts": {"scheduled": len(answers), "ok": 0, "empty": 0, "error": 0},
        "timing": {"started_at": _now(), "finished_at": None, "total_latency_ms": 0},
        "answers": [], "predictions": [], "metrics": None, "score_details": None,
    }
    report_path = runs_dir / f"{run_id}.json"
    models = set()
    try:
        for a in answers:
            inp = {k: a[k] for k in ("answer_id", "brand_id", "question", "ai_answer")}
            r = run_agent(inp, agent_version, provider)
            trace = runs_dir / "exchanges" / run_id / f"{a['answer_id']}.json"
            _write(trace, r.exchange)
            models.update(t["model"] for t in r.exchange["turns"] if t["model"])
            report["answers"].append({"answer_id": r.answer_id, "status": r.status, "error": r.error,
                                      "latency_ms": r.latency_ms, "claim_count": len(r.claims),
                                      "trace_ref": str(trace.relative_to(runs_dir.parent))})
            report["predictions"] += [{"answer_id": r.answer_id, **c.model_dump()} for c in r.claims]
            report["counts"][r.status] += 1
            report["timing"]["total_latency_ms"] += r.latency_ms
            if log:
                log(f"{r.answer_id} {r.status} {len(r.claims)} claims {r.latency_ms}ms {r.error or ''}")
        report["status"] = "completed"
    except BaseException as e:  # interrupted or infra failure: keep partial evidence, mark failed
        report["status"] = "failed"
        report["failure"] = {"type": type(e).__name__, "message": sanitize(e)}
        raise
    finally:
        report["resolved_model"] = sorted(models)[0] if len(models) == 1 else sorted(models) or None
        report["timing"]["finished_at"] = _now()
        _write(report_path, report)  # predictions saved before any gold is read
        if report["status"] == "completed":
            score_run(report)
            _write(report_path, report)
        if persist:
            persist(report)
    return report


def score_run(report):
    """Post-save scoring step: the only place the runner touches gold."""
    from app.scorer import score
    by_answer = {}
    for p in report["predictions"]:
        by_answer.setdefault(p["answer_id"], []).append({k: v for k, v in p.items() if k != "answer_id"})
    results = [{"answer_id": a["answer_id"], "status": a["status"], "error": a["error"],
                "claims": by_answer.get(a["answer_id"], [])} for a in report["answers"]]
    s = score(report["split"], [a["answer_id"] for a in report["answers"]], results)
    metric_keys = ("G", "P", "M", "C", "extraction", "joint", "coverage")
    report["metrics"] = {k: s[k] for k in metric_keys}
    report["score_details"] = {k: v for k, v in s.items() if k not in metric_keys + ("split", "counts")}
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app.runner")
    ap.add_argument("--version", choices=("v1", "v2"))
    ap.add_argument("--split", choices=SPLITS)
    args = ap.parse_args(argv)
    if not (args.version and args.split):
        ap.error("--version and --split are required")
    from app.providers import OpenAIProvider
    report = run_split(args.split, args.version, OpenAIProvider(), log=lambda m: print(m, flush=True))
    print(json.dumps({k: report[k] for k in ("run_id", "status", "counts", "metrics")}, indent=2))
    print(f"report: runs/{report['run_id']}.json")


if __name__ == "__main__":
    sys.exit(main())
