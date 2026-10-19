"""Output models and contract checks. Offsets come from segments, never from the model."""
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from app import data

Label = Literal["supported", "contradicted", "unknown"]


class OutputValidationError(Exception):
    pass


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Selection(_Strict):
    sentence_index: int
    label: Label
    fact_ids: list[str]
    reason: str


class Claim(_Strict):
    claim_id: str
    start_char: int
    end_char: int
    quote: str
    label: Label
    fact_ids: list[str]
    reason: str


class AgentOutput(_Strict):
    answer_id: str
    claims: list[Claim]


def _parse(model, raw):
    if isinstance(raw, model):
        return raw
    try:
        return model.model_validate(raw)
    except ValidationError as e:
        raise OutputValidationError(f"invalid {model.__name__}: {e.errors(include_url=False, include_input=False)}") from None


def build_claims(answer_id, ai_answer, segments, selections):
    """Turn model selections (sentence indices) into validated claims sorted by start_char."""
    if not isinstance(selections, list):
        raise OutputValidationError("selections must be a list")
    sels = [_parse(Selection, s) for s in selections]
    seen = set()
    claims = []
    for s in sels:
        if not 0 <= s.sentence_index < len(segments):
            raise OutputValidationError(f"sentence_index {s.sentence_index} out of range")
        if s.sentence_index in seen:
            raise OutputValidationError(f"sentence_index {s.sentence_index} repeated")
        seen.add(s.sentence_index)
        _, start, end, _ = segments[s.sentence_index]
        claims.append(Claim(claim_id=f"{answer_id}:{start}:{end}", start_char=start, end_char=end,
                            quote=ai_answer[start:end], label=s.label, fact_ids=s.fact_ids, reason=s.reason))
    claims.sort(key=lambda c: c.start_char)
    return validate_claims(ai_answer, claims, answer_id=answer_id)


def _too_many_sentences(reason):
    # ponytail: a new sentence = terminal punctuation, whitespace, capital letter; "vs. StoneWave" still overcounts
    return len(re.split(r"(?<=[.!?])\s+(?=[A-Z])", reason.strip())) > 2


def validate_claims(ai_answer, claims, answer_id=None, known_fact_ids=None):
    """Enforce every Agent-contract rule; return list[Claim] or raise OutputValidationError."""
    known = data.fact_ids() if known_fact_ids is None else known_fact_ids
    parsed = [_parse(Claim, c) for c in claims]
    ids = set()
    prev = None
    for c in parsed:
        s, e = c.start_char, c.end_char
        if not 0 <= s < e <= len(ai_answer):
            raise OutputValidationError(f"{c.claim_id}: span out of range")
        if c.quote != ai_answer[s:e]:
            raise OutputValidationError(f"{c.claim_id}: quote mismatch")
        if c.quote != c.quote.strip():
            raise OutputValidationError(f"{c.claim_id}: leading/trailing whitespace")
        prefix = c.claim_id.rsplit(":", 2)[0]
        answer_id = answer_id or prefix
        if c.claim_id != f"{answer_id}:{s}:{e}":
            raise OutputValidationError(f"{c.claim_id}: bad claim_id")
        if c.claim_id in ids:
            raise OutputValidationError(f"{c.claim_id}: duplicate claim_id")
        ids.add(c.claim_id)
        if prev and s < prev.start_char:
            raise OutputValidationError(f"{c.claim_id}: unsorted")
        if prev and s < prev.end_char:
            raise OutputValidationError(f"{c.claim_id}: overlapping")
        if bad := [f for f in c.fact_ids if f not in known]:
            raise OutputValidationError(f"{c.claim_id}: unknown fact_ids {bad}")
        if _too_many_sentences(c.reason):
            raise OutputValidationError(f"{c.claim_id}: reason over 2 sentences")
        prev = c
    return parsed
