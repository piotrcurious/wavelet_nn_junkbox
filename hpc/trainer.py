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
