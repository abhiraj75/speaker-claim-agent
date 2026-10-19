# AI assistance log
Tool: Claude Code. Agent provider: OpenAI, model gpt-5-mini (reasoning_effort=medium).

## How it was used
- I drove Claude Code phase by phase from `CLAUDE.md`, a project guide I wrote from the spec.
- Claude Code automated the repetitive implementation work: scaffolding, CSV loading, the validator, the agent loop, the scorer, the runner, the DB/API layer, and the tests.
- Claude Code ran the test suite and the live runs and prepared the failure analysis.

## What was mine
- The decisions and the logic behind them: the architecture and contract in `CLAUDE.md`, the model and settings, what to fix and what to defer, and the single targeted v2 change.
- Claude Code proposed options with evidence; I chose between them.

## What I checked
- Data hashes and segment offsets.
- The live exchanges, including the real `get_brand_facts` call and the absence of secrets.
- Agent output compared against the gold for A01.
- The v1 failure buckets and the v1-vs-v2 fixes and regressions.
- That every phase ended with the tests passing before it was committed.
