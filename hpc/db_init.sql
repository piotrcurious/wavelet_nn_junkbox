-- db_init.sql
-- Run on the Postgres server: psql -f db_init.sql

CREATE TABLE IF NOT EXISTS shards (
    shard_id SERIAL PRIMARY KEY,
    shard_path TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending, running, completed, failed
    claimed_by TEXT NULL,                    -- node id or worker id
    attempts INT NOT NULL DEFAULT 0,
    last_update TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS checkpoints (
    ckpt_id SERIAL PRIMARY KEY,
    shard_id INT REFERENCES shards(shard_id),
    node_id TEXT,
    path TEXT NOT NULL,
    metrics JSONB NULL,
    ts TIMESTAMP WITH TIME ZONE DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
    event_id SERIAL PRIMARY KEY,
    ts TIMESTAMP WITH TIME ZONE DEFAULT now(),
    event_type TEXT,
    payload JSONB
);

CREATE INDEX IF NOT EXISTS idx_shard_status ON shards(status);
