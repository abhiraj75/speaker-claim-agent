"""Model providers. send() returns {tool_calls:[{id,name,arguments}], content, model, usage}."""
import copy
import os
import re
from typing import Protocol

import openai
from dotenv import load_dotenv

SEED = 7
REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_SECRET_RE = re.compile(r"sk-[A-Za-z0-9_\-]+|req_[A-Za-z0-9]+|Bearer\s+\S+")


def redact(text):
    """Strip keys, bearer tokens and request IDs."""
    return _SECRET_RE.sub("[redacted]", str(text))


def sanitize(text):
    """Redacted, length-capped error message."""
    return redact(text)[:500]


class ProviderTimeout(Exception):
    pass


class ProviderError(Exception):
    pass


class Provider(Protocol):
    model: str
    settings: dict

    def send(self, messages, tools, tool_choice) -> dict: ...


class OpenAIProvider:
    def __init__(self):
        load_dotenv()
        self.model = os.environ["MODEL"]
        self.timeout = float(os.getenv("PROVIDER_TIMEOUT_S") or 60)
        if self.model.startswith(REASONING_PREFIXES):  # reasoning models reject temperature
            effort = os.getenv("REASONING_EFFORT") or None
            self.settings = {"reasoning_effort": effort, "seed": SEED} if effort else {"seed": SEED}
        else:
            self.settings = {"temperature": 0, "seed": SEED}
        self.client = openai.OpenAI(timeout=self.timeout, max_retries=0)

    def send(self, messages, tools, tool_choice):
        try:
            r = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice=tool_choice, **self.settings)
        except openai.APITimeoutError:
            raise ProviderTimeout(f"provider request exceeded {self.timeout:g}s") from None
        except openai.APIError as e:
            body = e.body if isinstance(e.body, dict) else {}
            status = getattr(e, "status_code", "")
            raise ProviderError(sanitize(f"{type(e).__name__} {status}: {body.get('message', '')}")) from None
        msg = r.choices[0].message
        if getattr(msg, "refusal", None):
            raise ProviderError("model refused")
        return {
            "tool_calls": [{"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
                           for c in msg.tool_calls or []],
            "content": msg.content,
            "model": r.model,
            "usage": r.usage.model_dump() if r.usage else None,
        }


class FakeProvider:
    """Replays scripted responses (dicts or exceptions to raise) and records every outgoing payload."""
    model = "fake-model"
    settings = {"seed": SEED}

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def send(self, messages, tools, tool_choice):
        self.calls.append(copy.deepcopy({"messages": messages, "tools": tools, "tool_choice": tool_choice}))
        r = self.script.pop(0)
        if isinstance(r, Exception):
            raise r
        return {"tool_calls": [], "content": None, "model": self.model, "usage": None, **r}
