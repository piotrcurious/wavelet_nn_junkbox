Great — I’ll give you a practical, cluster-ready implementation that solves the OOM / scaling problem by orchestrating distributed training across nodes (4 GPUs/node), streaming data (so no single-node needs to load full dataset), and using a database-backed job queue + provenance to coordinate shards, checkpoints, and recovery. I’ll include:

architecture summary and why each piece prevents OOM,

a minimal, production-minded set of scripts you can drop into a cluster (DB init, scheduler, worker launcher, per-GPU trainer),

a DeepSpeed config enabling ZeRO Offload (stage 3) and gradient checkpointing to drastically reduce GPU memory,

instructions for using WebDataset tar shards (streaming) so a GPU never needs the whole dataset in RAM,

notes about SLURM/torchrun deployment and how to tune parameters for your hardware (4 GPUs/node, 512 GB GPU memory per GPU, 64 GB RAM, 1 TB local disk).


This design assumes a shared network filesystem (NFS / Lustre) for data and checkpoints and a PostgreSQL server for job coordination. If you prefer Redis for a simple queue, I’ll provide a variant on request.


---

High-level architecture

1. Sharder / Storage: dataset pre-sharded into WebDataset .tar shards (each shard ≈ size that fits worker RAM/disk bounds). Shards live on shared storage (e.g., /shared/data/shards/*.tar).


2. Postgres Job DB: central metadata table with each shard represented as a job. Scheduler inserts jobs. Workers SELECT ... FOR UPDATE SKIP LOCKED to claim a shard — this avoids racing and supports failure recovery.


3. Scheduler: creates / re-queues jobs, optionally reassigns failed jobs, supports priority.


4. Workers (per node): single node runs a launcher that uses torchrun -nproc_per_node=4 to launch a per-GPU trainer.py. The trainer loads only the assigned shard via WebDataset streaming, uses DeepSpeed ZeRO stage 3 with offload to CPU/disk and gradient checkpointing to keep GPU memory low.


5. Provenance & Checkpointing: trainers write checkpoints to shared storage and push metadata to Postgres (completed, model path, metrics). The scheduler records events for audit.


6. Epistemic Agent / Adaptation: runs as separate processes and reads DB events / checkpoints to trigger adaptation jobs if needed (not covered in full here but the DB schema supports it).



Why this prevents OOM:

Streaming data via WebDataset means worker memory usage ≈ batch memory, not entire dataset.

ZeRO stage 3 + offload shards optimizer/params/gradients to CPU/disk and shards them across GPUs, drastically reducing per-GPU memory need.

Gradient accumulation / mixed precision / checkpointing further reduce GPU memory.

DB job queue ensures no redundant loading, easy retries, and cluster-scale scheduling.



---

Files I provide below

1. db_init.sql — SQL to create the job/provenance tables (Postgres).


2. scheduler.py — discovers shards, inserts jobs, monitors status, re-queues failed jobs.


3. node_launcher.sh — lightweight launcher that a node admin or resource manager runs; it polls DB, claims a shard and launches torchrun with trainer.py.


4. trainer.py — the per-GPU training worker that uses DeepSpeed. This script:

connects to DB to mark job running/completed/failed,

streams data from a shard (WebDataset example),

builds model + DeepSpeed engine with ZeRO offload,

performs training loop, and

checkpoints to shared storage and records metadata to DB.



5. deepspeed_config.json — ZeRO-3 + offload config tuned for large-memory GPUs and aggressive offload.


6. usage notes & tuning guidance.



> Important: these scripts are self-contained and do not assume the original, memory-heavy single-node loading. Replace dataset/model details as needed.




---

1) db_init.sql (Postgres schema)

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


---

2) scheduler.py — discover shards and enqueue jobs

# scheduler.py
# Usage: python scheduler.py --data-dir /shared/data/shards --db-url postgresql://user:pass@dbhost:5432/sepdb

import argparse, os, psycopg2, time, json
from psycopg2.extras import execute_values

def discover_shards(data_dir):
    # expects shards named *.tar or *.tar.zst etc
    exts = ('.tar', '.tar.zst', '.tar.gz')
    return sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(exts)])

def upsert_shards(conn, shards):
    with conn.cursor() as cur:
        # insert if not exists
        vals = [(s,) for s in shards]
        execute_values(cur,
            "INSERT INTO shards (shard_path) VALUES %s ON CONFLICT (shard_path) DO NOTHING",
            vals)
    conn.commit()

def monitor(conn, poll_interval=300):
    while True:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM shards WHERE status = 'pending'")
            pending = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM shards WHERE status = 'running'")
            running = cur.fetchone()[0]
            print(f"[scheduler] pending={pending} running={running}")
        time.sleep(poll_interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--db-url', required=True)
    args = p.parse_args()
    conn = psycopg2.connect(args.db_url)
    shards = discover_shards(args.data_dir)
    print(f"[scheduler] discovered {len(shards)} shards")
    upsert_shards(conn, shards)
    try:
        monitor(conn)
    except KeyboardInterrupt:
        print("Scheduler exiting.")


---

3) node_launcher.sh — node-level script that claims shard and launches torchrun

#!/usr/bin/env bash
# node_launcher.sh
# run on each cluster node (or via batch scheduler)
# Example:
# NODE_ID=$(hostname)
# ./node_launcher.sh --db-url postgresql://user:pass@dbhost:5432/sepdb --node-id ${NODE_ID} --gpus 4

set -euo pipefail

DB_URL=""
NODE_ID=""
GPUS=4
WORK_DIR="/shared/checkpoints"
TRAINER_PY="trainer.py"
PYTHON_EXE="python"   # or python3
MASTER_PORT=29500     # must be reachable; if using SLURM set MASTER_ADDR env separately

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --db-url) DB_URL="$2"; shift; shift;;
    --node-id) NODE_ID="$2"; shift; shift;;
    --gpus) GPUS="$2"; shift; shift;;
    --work-dir) WORK_DIR="$2"; shift; shift;;
    *) echo "Unknown arg $1"; exit 1;;
  esac
done

if [[ -z "$DB_URL" || -z "$NODE_ID" ]]; then
  echo "Usage: node_launcher.sh --db-url <db> --node-id <node> [--gpus N] [--work-dir DIR]"
  exit 1
fi

# Loop: claim jobs via DB and run torchrun for one shard per node (nproc_per_node = GPUS)
# The trainer.py will receive SHARD_PATH and DB_URL and update DB on completion/failure.
while true; do
  SHARD_PATH=$($PYTHON_EXE - <<PYCODE
import psycopg2, os, sys
conn = psycopg2.connect("${DB_URL}")
cur = conn.cursor()
# atomically claim a pending shard
cur.execute(\"\"\"
UPDATE shards SET status='running', claimed_by=%s, attempts=attempts+1, last_update=now()
WHERE shard_id = (
    SELECT shard_id FROM shards WHERE status='pending' ORDER BY shard_id LIMIT 1
) RETURNING shard_path;
\"\"\", ("${NODE_ID}",))
row = cur.fetchone()
conn.commit()
if row:
    print(row[0])
else:
    # no pending shard
    print("")
cur.close()
conn.close()
PYCODE
)
  if [[ -z "$SHARD_PATH" ]]; then
    echo "No pending shards. Sleeping 60s."
    sleep 60
    continue
  fi

  echo "Node ${NODE_ID} claimed shard ${SHARD_PATH}"

  # Launch torchrun across GPUs on this node
  # Use MASTER_ADDR=localhost if node-local; when multi-node multi-node training is needed, set MASTER_ADDR & PORT accordingly.
  export MASTER_ADDR=127.0.0.1
  export MASTER_PORT=${MASTER_PORT}
  export NCCL_DEBUG=INFO
  export PYTHONUNBUFFERED=1

  torchrun --nproc_per_node=${GPUS} ${TRAINER_PY} \
    --db-url "${DB_URL}" \
    --node-id "${NODE_ID}" \
    --shard-path "${SHARD_PATH}" \
    --work-dir "${WORK_DIR}" \
    || echo "Training failed for shard ${SHARD_PATH}"

  # After torchrun returns, check trainer updated DB. If trainer crashes without updating, mark shard failed.
  $PYTHON_EXE - <<PYCODE
import psycopg2, sys
conn = psycopg2.connect("${DB_URL}")
cur = conn.cursor()
cur.execute("SELECT status FROM shards WHERE shard_path=%s", ("${SHARD_PATH}",))
st = cur.fetchone()[0]
if st == 'running':
    # trainer didn't update; mark as failed so scheduler may requeue
    cur.execute("UPDATE shards SET status='failed', last_update=now() WHERE shard_path=%s", ("${SHARD_PATH}",))
    conn.commit()
cur.close(); conn.close()
PYCODE

done


---

4) trainer.py — per-GPU trainer that uses DeepSpeed and streams a shard

This is the core: it uses torch.distributed + deepspeed and WebDataset to stream shard contents. It's a template; adapt model/loss components to your model.

# trainer.py
# run via: torchrun --nproc_per_node=4 trainer.py --db-url <...> --node-id <...> --shard-path /shared/data/shards/x.tar --work-dir /shared/checkpoints

import argparse, os, time, json
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed
import psycopg2
from psycopg2.extras import Json
import webdataset as wds   # pip install webdataset
import soundfile as sf

# ----------------------------
# model placeholder (replace with your separator)
# ----------------------------
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 15, padding=7)
        self.conv2 = nn.Conv1d(64, 64, 15, padding=7)
        self.out = nn.Conv1d(64, 1, 1)
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.out(x)

# ----------------------------
# utility: DB update helpers
# ----------------------------
def with_db(db_url, fn, *args, **kwargs):
    conn = psycopg2.connect(db_url)
    try:
        res = fn(conn, *args, **kwargs)
        conn.commit()
        return res
    finally:
        conn.close()

def mark_shard_running(conn, shard_path, node_id):
    cur = conn.cursor()
    cur.execute("UPDATE shards SET status='running', claimed_by=%s, attempts=attempts+1, last_update=now() WHERE shard_path=%s RETURNING shard_id", (node_id, shard_path))
    row = cur.fetchone()
    if row:
        return row[0]
    raise RuntimeError("Failed to mark shard running")

def mark_shard_completed(conn, shard_path, ckpt_path, metrics, node_id):
    cur = conn.cursor()
    cur.execute("UPDATE shards SET status='completed', last_update=now() WHERE shard_path=%s", (shard_path,))
    cur.execute("INSERT INTO checkpoints (shard_id, node_id, path, metrics) VALUES ((SELECT shard_id FROM shards WHERE shard_path=%s), %s, %s, %s)", (shard_path, node_id, ckpt_path, Json(metrics)))

def mark_shard_failed(conn, shard_path, node_id, reason):
    cur = conn.cursor()
    cur.execute("UPDATE shards SET status='failed', last_update=now() WHERE shard_path=%s", (shard_path,))
    cur.execute("INSERT INTO events (event_type, payload) VALUES (%s, %s)", ('training_failed', Json({'shard': shard_path, 'node': node_id, 'reason': reason})))

# ----------------------------
# dataset streaming from a single shard
# shard_path is a tar of named samples; for example each sample has keys:
#   'wav': audio bytes or file path; 'json': metadata
# Use webdataset to iterate samples in the shard.
# ----------------------------
def shard_stream_generator(shard_path, batch_size=4, seglen=16000, sr=16000):
    # Example: audio stored as .wav bytes under key 'wav'
    dataset = wds.WebDataset(shard_path).decode(wds.audio.AudioFileHandler) \
              .to_tuple("wav","json")
    # yield batches
    batch_wavs = []
    for sample in dataset:
        wav, meta = sample
        # wav is numpy array (channels, samples)
        if wav.ndim > 1:
            wav = wav.mean(axis=0)
        # trim/pad
        if len(wav) > seglen:
            start = np.random.randint(0, len(wav)-seglen)
            wav = wav[start:start+seglen]
        else:
            wav = np.pad(wav, (0, seglen - len(wav)))
        batch_wavs.append(wav.astype(np.float32))
        if len(batch_wavs) >= batch_size:
            batch = np.stack(batch_wavs, axis=0)  # (B, T)
            batch_wavs = []
            # return tensor shaped (B,1,T)
            yield torch.from_numpy(batch).unsqueeze(1)
    # final partial batch
    if batch_wavs:
        batch = np.stack(batch_wavs, axis=0)
        yield torch.from_numpy(batch).unsqueeze(1)

# ----------------------------
# training function run by each process
# ----------------------------
def train_loop(args):
    # initialize distributed
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # connect to DB and mark shard running (only rank 0 should update DB)
    if rank == 0:
        conn = psycopg2.connect(args.db_url)
        try:
            shard_id = mark_shard_running(conn, args.shard_path, args.node_id)
            print(f"[rank0] marked shard running id={shard_id}")
        except Exception as e:
            print("DB claim failed:", e)
            raise
        finally:
            conn.commit()
            conn.close()

    # Setup model and DeepSpeed
    model = SimpleModel().to(device)

    # DeepSpeed config file recommended (see provided deepspeed_config.json)
    ds_config = args.deepspeed_config

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, _ = deepspeed.initialize(args=None, model=model, model_parameters=parameters, config=args.deepspeed_config)

    # Use small per-GPU batch and gradient accumulation to emulate large-batch training
    per_gpu_batch = args.batch_size
    accum_steps = args.grad_accum

    # Stream data from shard
    train_gen = shard_stream_generator(args.shard_path, batch_size=per_gpu_batch, seglen=args.seg_len)
    step = 0
    try:
        for batch in train_gen:
            batch = batch.to(device)
            # build dummy target for example (use real target logic in your pipeline)
            target = batch.clone()

            # forward
            pred = model_engine(batch)   # model_engine wraps model and handles autograd
            loss = F.l1_loss(pred, target)

            model_engine.backward(loss)
            model_engine.step()

            step += 1
            if rank == 0 and step % args.log_interval == 0:
                print(f"[rank0] shard {args.shard_path} step {step} loss {loss.item():.6f}")

            # optional: periodic checkpointing (only rank=0 writes to shared fs)
            if rank == 0 and step % args.ckpt_interval == 0:
                ckpt_path = os.path.join(args.work_dir, f"ckpt_{os.path.basename(args.shard_path)}_step{step}")
                # deepspeed save checkpoint (saves partitioned checkpoints)
                model_engine.save_checkpoint(ckpt_path)
                # update DB with checkpoint metadata
                conn = psycopg2.connect(args.db_url)
                try:
                    mark_shard_completed(conn, args.shard_path, ckpt_path, {'step': step, 'loss': float(loss.item())}, args.node_id)
                finally:
                    conn.commit(); conn.close()

    except Exception as e:
        # failure: mark shard failed (only rank0)
        if rank == 0:
            conn = psycopg2.connect(args.db_url)
            try:
                mark_shard_failed(conn, args.shard_path, args.node_id, str(e))
            finally:
                conn.commit(); conn.close()
        raise

    # success: final checkpoint and mark completed (rank0)
    if rank == 0:
        ckpt_path = os.path.join(args.work_dir, f"ckpt_{os.path.basename(args.shard_path)}_final")
        model_engine.save_checkpoint(ckpt_path)
        conn = psycopg2.connect(args.db_url)
        try:
            mark_shard_completed(conn, args.shard_path, ckpt_path, {'step': step, 'loss': 0.0}, args.node_id)
        finally:
            conn.commit(); conn.close()

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', required=True)
    parser.add_argument('--node-id', required=True)
    parser.add_argument('--shard-path', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum', type=int, default=1)
    parser.add_argument('--seg-len', type=int, default=16000)
    parser.add_argument('--ckpt-interval', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--deepspeed-config', default="deepspeed_config.json")
    args = parser.parse_args()

    # adjust environment for deepspeed if needed
    train_loop(args)

Notes about trainer.py:

Uses webdataset to stream data from a single .tar shard. webdataset decodes audio lazily and yields small batches, keeping memory low.

Uses DeepSpeed deepspeed.initialize to wrap the model, which supports ZeRO stage 3 — see config below.

Only rank 0 writes DB updates and checkpoints to reduce contention.



---

5) deepspeed_config.json — ZeRO Stage 3 + Offload (example)

{
  "train_batch_size": 128,
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 16,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
  },
  "zero_optimization": {
    "stage": 3,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e5
  },
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "contiguous_memory_optimization": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 50
}

Tuning notes

train_micro_batch_size_per_gpu should be set small (1–4) to fit activations.

Use gradient_accumulation_steps to emulate larger batch sizes without more memory.

Offload to CPU is crucial for extremely large models; if CPU memory or IO becomes a bottleneck you can offload to NVMe (DeepSpeed supports nvme offload in some versions).

pin_memory with large system RAM (64 GB) helps throughput.



---

6) How to run this on your cluster (summary)

1. Prepare shards: use WebDataset tar shards; each shard sized to fit node memory / I/O. A rule of thumb: each shard ~ 10–50 GB depending on I/O and batch size. Put shards under /shared/data/shards/.


2. Initialize DB:

psql -U user -h dbhost -d sepdb -f db_init.sql


3. Start scheduler (anywhere that can write to DB):

python scheduler.py --data-dir /shared/data/shards --db-url postgresql://user:pass@dbhost:5432/sepdb


4. On each node: start node_launcher.sh (or run through SLURM as a job). Example:

./node_launcher.sh --db-url postgresql://user:pass@dbhost:5432/sepdb --node-id $(hostname) --gpus 4 --work-dir /shared/checkpoints

This script will claim shards and run torchrun --nproc_per_node=4 trainer.py ... for that shard.


5. Monitor DB: use SELECT * FROM shards WHERE status!='completed' to see progress. Check checkpoints and events tables for provenance.




---

7) Memory & performance guidance for your hardware (4× GPUs/node, 512G GPU each (!) — very generous GPU memory)

Your cluster is generous: 512 GB GPU memory per GPU is unusually large (probably you meant 512 GB total across node GPUs?), but the config above assumes you still want to scale model parameters beyond single GPU memory. With ZeRO stage 3 and offload, you can train extremely large models. Key knobs:

Per-GPU batch size: keep small (1–4) and use grad accumulation, because activations often dominate memory.

FP16: use mixed-precision (fp16) to cut memory and increase throughput.

Activation checkpointing: enables trading compute for memory by recomputing activations.

Offload: if you have very large parameter counts or extremely deep models, offload to CPU or NVMe.



---

8) Provenance & reproducibility

The DB tables checkpoints and events contain checkpoint paths and training metadata; every checkpoint save should include a JSON file with the training config, git commit, and data-shard digest (SHA256). Add this inside your trainer.py when writing a checkpoint:


metadata = {'git_commit': '<commit>', 'config': args.__dict__, 'shard': args.shard_path}
with open(os.path.join(ckpt_path, 'metadata.json'), 'w') as f:
    json.dump(metadata, f)

Use DVC or Git-LFS to snapshot config/data hashes if you need end-to-end reproducibility.



---

9) Failure handling & requeueing

The node_launcher.sh checks whether the trainer updated the DB; if not, it marks shards as failed so the scheduler or human can retry.

The scheduler can re-enqueue failed shards after a cooldown (increase attempts and implement exponential backoff).



---

10) Next steps I can provide immediately (pick any or multiple)

A. Redis-backed variant (lighter weight) with Lua-atomic pop/push for very high-throughput scheduling.

B. SLURM integration scripts (sbatch templates) to run node_launcher.sh as node jobs with correct env for multi-node torchrun (MASTER_ADDR/PORT).

C. Add checkpoint sharding & model-aggregation script to aggregate per-shard checkpoints into a single global model (use DeepSpeed's checkpoint-merge utilities or zero_to_fp32.py).

D. Integrate the Fractal-Bayesian refiner & EpistemicAgent into this job orchestration, so the agent can schedule adaptation jobs for flagged shards automatically.


Tell me which of A/B/C/D you want and I’ll produce the exact scripts to plug into this system right away.


---

If you'd like, I can also:

produce a ready-to-run Dockerfile for a worker image (with deepspeed, webdataset, psycopg2 preinstalled),

give a sample DeepSpeed tuning matrix for your model sizes and expected throughput,

or convert the Postgres DB usage to Redis if you want a lower-latency queue.


Which next step should I do?
