"""Load answers and brand facts from the untouched CSVs in data/."""
import csv
import hashlib
from datetime import date
from functools import lru_cache
from pathlib import Path

DATASET_VERSION = "long-brand-answers-50-v4"
FACT_VERSION = "speaker-catalogue-v4"
EVALUATOR_VERSION = "sentence-spans-and-labels-v4"
QUESTION_DATE = "2026-10-01"
BRANDS = ("cleartrail", "stonewave", "cloudnote")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EXPECTED_HASHES = {
    "answers.csv": "fa28e813ea54b4e5ea985585f22f084d826241e1182accbc27ace17066125724",
    "brand_facts.csv": "2190d40fad2dd167ef2437956c43d595aaf2554ea673a4789b9343cb7e2936f2",
    "development_gold.csv": "afe84cc78f304afb6a4aaa505dd7504af80fcbf8bc507806aa5ff0aab6bf4850",
}


def read_csv(name):
    with open(DATA_DIR / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def file_hashes():
    return {name: hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest() for name in EXPECTED_HASHES}


def parse_value(value, value_type):
    if value_type == "number":
        return int(value) if value.lstrip("-").isdigit() else float(value)
    if value_type == "boolean":
        if value not in ("true", "false"):
            raise ValueError(f"bad boolean {value!r}")
        return value == "true"
    if value_type == "string":
        return value
    raise ValueError(f"bad value_type {value_type!r}")


@lru_cache
def load_answers():
    return read_csv("answers.csv")


@lru_cache
def load_facts():
    rows = []
    for r in read_csv("brand_facts.csv"):
        for k in ("valid_from", "valid_to"):
            date.fromisoformat(r[k])  # reject malformed dates
        rows.append({**r, "value": parse_value(r["value"], r["value_type"]),
                     "unit": r["unit"] or None, "condition": r["condition"] or None})
    return rows


@lru_cache
def facts_by_brand():
    out = {b: [] for b in BRANDS}
    for r in load_facts():
        out[r["brand_id"]].append(r)
    return out


@lru_cache
def fact_ids():
    return frozenset(r["fact_id"] for r in load_facts())
