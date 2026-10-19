"""Strict tool schemas and the allowlisted executor. Never executes names taken from answer text."""
import json

from app.data import BRANDS, FACT_VERSION, facts_by_brand

GET_BRAND_FACTS = {"type": "function", "function": {
    "name": "get_brand_facts",
    "description": "Return the catalogue fact rows for one known brand.",
    "parameters": {
        "type": "object",
        "properties": {"brand_id": {"type": "string", "enum": list(BRANDS)}},
        "required": ["brand_id"],
        "additionalProperties": False,
    },
    "strict": True,
}}

SUBMIT_CLAIMS = {"type": "function", "function": {
    "name": "submit_claims",
    "description": "Submit the final factual claim sentences, by sentence index, with labels. Call exactly once.",
    "parameters": {
        "type": "object",
        "properties": {"claims": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "sentence_index": {"type": "integer"},
                "label": {"type": "string", "enum": ["supported", "contradicted", "unknown"]},
                "fact_ids": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["sentence_index", "label", "fact_ids", "reason"],
            "additionalProperties": False,
        }}},
        "required": ["claims"],
        "additionalProperties": False,
    },
    "strict": True,
}}

TOOLS = [GET_BRAND_FACTS, SUBMIT_CLAIMS]
ALLOWED_TOOLS = {"get_brand_facts", "submit_claims"}


class ToolError(Exception):
    def __init__(self, type, message):
        super().__init__(message)
        self.type = type
        self.message = message


def get_brand_facts(brand_id):
    if brand_id not in BRANDS:
        raise ToolError("unknown_brand", f"unknown brand {str(brand_id)[:40]!r}")
    return {"fact_version": FACT_VERSION, "brand_id": brand_id, "facts": facts_by_brand()[brand_id]}


def execute(name, arguments):
    """Run a lookup tool call. submit_claims is terminal and handled by the agent, not here."""
    if name != "get_brand_facts":
        raise ToolError("illegal_tool", f"tool {str(name)[:40]!r} is not executable")
    try:
        args = json.loads(arguments)
    except (TypeError, ValueError):
        raise ToolError("invalid_output", "get_brand_facts arguments are not valid JSON") from None
    if not isinstance(args, dict) or set(args) != {"brand_id"}:
        raise ToolError("invalid_output", "get_brand_facts takes exactly one argument: brand_id")
    return get_brand_facts(args["brand_id"])
