CREATE TABLE IF NOT EXISTS runs (
    id                uuid PRIMARY KEY,
    split             text NOT NULL CHECK (split IN ('development', 'check')),
    agent_version     text NOT NULL CHECK (agent_version IN ('v1', 'v2')),
    prompt_version    text NOT NULL,
    dataset_version   text NOT NULL,
    fact_version      text NOT NULL,
    evaluator_version text NOT NULL,
    provider          text NOT NULL,
    model             text,
    settings          jsonb NOT NULL,
    file_hashes       jsonb NOT NULL,
    source_revision   text,
    status            text NOT NULL CHECK (status IN ('completed', 'failed')),
    counts            jsonb NOT NULL,
    metrics           jsonb,
    score_details     jsonb,
    created_at        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS answer_results (
    id         bigserial PRIMARY KEY,
    run_id     uuid NOT NULL REFERENCES runs (id) ON DELETE CASCADE,
    answer_id  text NOT NULL,
    status     text NOT NULL CHECK (status IN ('ok', 'empty', 'error')),
    error      jsonb,
    latency_ms integer,
    trace_ref  text,
    UNIQUE (run_id, answer_id),
    CHECK ((status = 'error') = (error IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS claim_results (
    id         bigserial PRIMARY KEY,
    run_id     uuid NOT NULL,
    answer_id  text NOT NULL,
    claim_id   text NOT NULL,
    start_char integer NOT NULL CHECK (start_char >= 0),
    end_char   integer NOT NULL,
    quote      text NOT NULL,
    label      text NOT NULL CHECK (label IN ('supported', 'contradicted', 'unknown')),
    fact_ids   text[] NOT NULL,
    reason     text NOT NULL,
    dev_match  jsonb,
    UNIQUE (run_id, answer_id, start_char, end_char),
    FOREIGN KEY (run_id, answer_id) REFERENCES answer_results (run_id, answer_id) ON DELETE CASCADE,
    CHECK (end_char > start_char)
);
