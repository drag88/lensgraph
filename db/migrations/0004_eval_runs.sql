-- 0004_eval_runs.sql — populated by phase 2

CREATE TABLE eval_runs (
    run_id              text PRIMARY KEY,
    run_date            date NOT NULL,
    code_path           text NOT NULL,
    chunking_strategy   text NOT NULL,
    embedding_model_id  text NOT NULL,
    generator_model_id  text,
    judge_model_id      text,
    judge_prompt_hash   text,
    candidate_set_yaml  jsonb NOT NULL,
    summary             jsonb,
    created_at          timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE eval_results (
    eval_run_id   text NOT NULL REFERENCES eval_runs(run_id) ON DELETE CASCADE,
    example_id    text NOT NULL,
    system_output jsonb NOT NULL,
    metrics       jsonb NOT NULL,
    PRIMARY KEY (eval_run_id, example_id)
);
