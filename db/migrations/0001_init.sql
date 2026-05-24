-- 0001_init.sql — talks + per-step status

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgmq;

CREATE TABLE schema_migrations (
    version    text PRIMARY KEY,
    applied_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE talks (
    video_id            text PRIMARY KEY,
    source_video_id     text,
    source_start_sec    real,
    source_end_sec      real,
    title               text NOT NULL,
    speaker             text NOT NULL,
    url                 text NOT NULL,
    duration_sec        integer NOT NULL CHECK (duration_sec > 0),
    format_tags         text[] NOT NULL,
    license             text NOT NULL,
    captions_source     text NOT NULL,
    transcript_path     text NOT NULL,
    transcript_sha256   text NOT NULL CHECK (transcript_sha256 ~ '^[a-f0-9]{64}$'),
    accessed_at         timestamptz NOT NULL,
    notes               text,
    ingested_at         timestamptz
);

CREATE TABLE ingest_step_status (
    video_id       text NOT NULL REFERENCES talks(video_id) ON DELETE CASCADE,
    step           text NOT NULL,
    entity_id      bigint NOT NULL DEFAULT 0,
    status         text NOT NULL,
    attempts       integer NOT NULL DEFAULT 0,
    started_at     timestamptz,
    completed_at   timestamptz,
    error_payload  jsonb,
    updated_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (video_id, step, entity_id)
);
CREATE INDEX ingest_step_status_video_idx ON ingest_step_status (video_id, status);
CREATE INDEX ingest_step_status_pending_idx ON ingest_step_status (step, status)
    WHERE status IN ('pending', 'in_progress', 'failed');
