-- 0003_frames.sql — visual retrieval substrate (now a true 5th channel)

CREATE TABLE frames (
    frame_id          bigserial PRIMARY KEY,
    video_id          text NOT NULL REFERENCES talks(video_id) ON DELETE CASCADE,
    frame_sec         real NOT NULL,
    image_path        text NOT NULL,
    sha256            text NOT NULL,
    pooled_embedding  vector(128),
    UNIQUE (video_id, frame_sec)
);
CREATE INDEX frames_video_idx ON frames (video_id, frame_sec);
CREATE INDEX frames_pooled_hnsw ON frames
    USING hnsw (pooled_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    WHERE pooled_embedding IS NOT NULL;

CREATE TABLE frame_patches (
    frame_id     bigint NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
    patch_index  integer NOT NULL,
    embedding    vector(128) NOT NULL,
    PRIMARY KEY (frame_id, patch_index)
);
CREATE INDEX frame_patches_frame_idx ON frame_patches (frame_id);
