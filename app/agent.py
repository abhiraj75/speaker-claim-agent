"""Claim agent: at most 3 model turns, real tool calls, Python-built offsets."""
import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.providers import ProviderError, ProviderTimeout, redact, sanitize
from app.segment import segment
from app.tools import ALLOWED_TOOLS, TOOLS, ToolError, execute
from app.validate import Claim, OutputValidationError, build_claims

MAX_TURNS = 3
TOOL_CHOICE = ["required", "auto", {"type": "function", "function": {"name": "submit_claims"}}]
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
AGENT_VERSIONS = ("v1", "v2")


class AnswerResult(BaseModel):
    answer_id: str
    status: Literal["ok", "empty", "error"]
    claims: list[Claim]
    error: dict | None
    latency_ms: int
    exchange: dict


class AgentError(Exception):
    def __init__(self, type, message):
        super().__init__(message)
        self.type = type
        self.message = message


def load_prompt(agent_version):
    if agent_version not in AGENT_VERSIONS:
        raise ValueError(f"unknown agent_version {agent_version!r}")
    text = (PROMPTS_DIR / f"{agent_version}.md").read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"prompts/{agent_version}.md is empty")
    return text


def user_message(inp, segments):
    # Neutralise a closing tag inside answer text so it cannot end the untrusted block early.
    lines = "\n".join(f"[{i}] {t}".replace("</untrusted_answer", "<\\/untrusted_answer") for i, _, _, t in segments)
    return (f"answer_id: {inp['answer_id']}\nbrand_id: {inp['brand_id']}\nquestion: {inp['question']}\n"
            f"<untrusted_answer>\n{lines}\n</untrusted_answer>")


def _loop(inp, segments, messages, provider, turns):
    for turn in range(MAX_TURNS):
        try:
            resp = provider.send(messages, TOOLS, TOOL_CHOICE[turn])
        except ProviderTimeout as e:
            raise AgentError("timeout", str(e)) from None
        except ProviderError as e:
            raise AgentError("provider_error", str(e)) from None
        calls = resp["tool_calls"]
        turns.append({"turn": turn + 1, "tool_choice": TOOL_CHOICE[turn], "model": resp["model"],
                      "usage": resp["usage"], "tool_calls": [c["name"] for c in calls]})
        assistant = {"role": "assistant", "content": resp["content"]}
        if calls:
            assistant["tool_calls"] = [{"id": c["id"], "type": "function",
                                        "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
        messages.append(assistant)
        if illegal := [c["name"] for c in calls if c["name"] not in ALLOWED_TOOLS]:
            raise AgentError("illegal_tool", f"tool {str(illegal[0])[:40]!r} is not allowed")
        submits = [c for c in calls if c["name"] == "submit_claims"]
        if len(submits) > 1:
            raise AgentError("invalid_output", "submit_claims called more than once")
        if submits:
            try:
                args = json.loads(submits[0]["arguments"])
            except (TypeError, ValueError):
                raise AgentError("invalid_output", "submit_claims arguments are not valid JSON") from None
            if not isinstance(args, dict) or set(args) != {"claims"}:
                raise AgentError("invalid_output", "submit_claims takes exactly one argument: claims")
            try:
                return build_claims(inp["answer_id"], inp["ai_answer"], segments, args["claims"])
            except OutputValidationError as e:
                raise AgentError("invalid_output", str(e)) from None
        for c in calls:
            try:
                result = execute(c["name"], c["arguments"])
            except ToolError as e:
                raise AgentError(e.type, e.message) from None
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
    raise AgentError("turn_limit", f"no valid submit_claims within {MAX_TURNS} turns")


def run_agent(inp, agent_version, provider):
    """inp: {answer_id, brand_id, question, ai_answer}. Any failure -> status error with zero claims."""
    start = time.perf_counter()
    segments = segment(inp["ai_answer"])
    messages = [{"role": "system", "content": load_prompt(agent_version)},
                {"role": "user", "content": user_message(inp, segments)}]
    turns = []
    claims, error = [], None
    try:
        claims = _loop(inp, segments, messages, provider, turns)
    except AgentError as e:
        error = {"type": e.type, "message": sanitize(e.message)}
    status = "error" if error else ("ok" if claims else "empty")
    exchange = {"answer_id": inp["answer_id"], "agent_version": agent_version, "requested_model": provider.model,
                "settings": provider.settings, "status": status, "error": error, "turns": turns, "messages": messages}
    exchange = json.loads(redact(json.dumps(exchange)))
    return AnswerResult(answer_id=inp["answer_id"], status=status, claims=claims, error=error,
                        latency_ms=round((time.perf_counter() - start) * 1000), exchange=exchange)
