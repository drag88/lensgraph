-- 0002_chunks_embeds.sql — text retrieval substrate

CREATE TABLE chunks (
    chunk_id            bigserial PRIMARY KEY,
    video_id            text NOT NULL REFERENCES talks(video_id) ON DELETE CASCADE,
    chunking_strategy   text NOT NULL,
    start_sec           real NOT NULL,
    end_sec             real NOT NULL CHECK (end_sec > start_sec),
    text                text NOT NULL,
    token_count         integer NOT NULL CHECK (token_count <= 512),
    frame_secs          real[] NOT NULL DEFAULT '{}',
    tsv                 tsvector GENERATED ALWAYS AS
                          (to_tsvector('english', text)) STORED,
    created_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (video_id, chunking_strategy, start_sec, end_sec)
);
CREATE INDEX chunks_tsv_idx ON chunks USING GIN (tsv);
CREATE INDEX chunks_video_strategy_idx ON chunks (video_id, chunking_strategy);

CREATE TABLE dense_embeds (
    chunk_id   bigint PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding  vector(1024) NOT NULL
);
CREATE INDEX dense_embeds_hnsw ON dense_embeds
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE sparse_embeds (
    chunk_id   bigint PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    embedding  sparsevec(250002) NOT NULL
);
CREATE INDEX sparse_embeds_hnsw ON sparse_embeds
    USING hnsw (embedding sparsevec_ip_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE chunk_token_embeds (
    chunk_id   bigint NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    position   integer NOT NULL,
    embedding  vector(1024) NOT NULL,
    PRIMARY KEY (chunk_id, position)
);
CREATE INDEX chunk_token_embeds_chunk_idx ON chunk_token_embeds (chunk_id);
