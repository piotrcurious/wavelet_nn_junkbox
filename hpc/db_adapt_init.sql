-- db_adapt_init.sql
-- Add adaptation_jobs and a more detailed checkpoints table fields.

CREATE TABLE IF NOT EXISTS adaptation_jobs (
    adapt_id SERIAL PRIMARY KEY,
    created_ts TIMESTAMP WITH TIME ZONE DEFAULT now(),
    checkpoint_path TEXT NOT NULL,         -- checkpoint to adapt
    requested_by TEXT NULL,                -- agent id or user
    status TEXT NOT NULL DEFAULT 'pending', -- pending, running, completed, failed
    claimed_by TEXT NULL,
    params JSONB NULL,                     -- adaptation config: lr, steps, dataset shard list...
    last_update TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- optional: add columns to checkpoints for sample counts / metadata
ALTER TABLE checkpoints
  ADD COLUMN IF NOT EXISTS examples_processed INT DEFAULT 0,
  ADD COLUMN IF NOT EXISTS epoch INT DEFAULT 0;
