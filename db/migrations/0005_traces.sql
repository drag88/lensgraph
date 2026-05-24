-- 0005_traces.sql — LangGraph observability (Postgres-native, ADR 002 honored)

CREATE TABLE traces (
    trace_id           text PRIMARY KEY,
    query              text NOT NULL,
    corpus_id          text NOT NULL,
    generator_model_id text,
    started_at         timestamptz NOT NULL DEFAULT now(),
    ended_at           timestamptz,
    status             text NOT NULL,
    final_answer       text,
    final_citations    jsonb,
    iterations         integer NOT NULL DEFAULT 0,
    latency_ms         integer
);
CREATE INDEX traces_started_idx ON traces (started_at DESC);

CREATE TABLE trace_spans (
    span_id        text PRIMARY KEY,
    trace_id       text NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
    parent_span_id text,
    node_name      text NOT NULL,
    iteration      integer NOT NULL DEFAULT 0,
    started_at     timestamptz NOT NULL,
    ended_at       timestamptz,
    input          jsonb,
    output         jsonb,
    metadata       jsonb,
    status         text NOT NULL
);
CREATE INDEX trace_spans_trace_idx ON trace_spans (trace_id, started_at);
CREATE INDEX trace_spans_node_idx ON trace_spans (node_name, started_at);
