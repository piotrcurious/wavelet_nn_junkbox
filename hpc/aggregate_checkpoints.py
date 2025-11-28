#!/usr/bin/env python3
# aggregate_checkpoints.py
# Usage:
#   python aggregate_checkpoints.py --db-url postgresql://user:pass@db:5432/sepdb \
#       --out /shared/models/agg_model.pt --filter "shard_like='%shard-prefix%'" \
#       [--finetune] [--val-shards /shared/data/val/*.tar] [--deepspeed-convert]

import argparse, os, json, tempfile, glob
import torch
import psycopg2
from psycopg2.extras import Json

def fetch_checkpoints(conn, filter_clause=None):
    cur = conn.cursor()
    if filter_clause:
        q = f"SELECT path, metrics, examples_processed FROM checkpoints WHERE path LIKE %s"
        cur.execute(q, (filter_clause,))
    else:
        cur.execute("SELECT path, metrics, examples_processed FROM checkpoints ORDER BY ts DESC")
    rows = cur.fetchall()
    cur.close()
    return rows

def load_state_dict_maybe_deepspeed(ckpt_path, try_deepspeed=True):
    """
    Try to load a standard torch state_dict; if it looks like a deepspeed partitioned checkpoint,
    either call DeepSpeed's convert helper (if available) or attempt to find a single 'pytorch_model.bin' in the checkpoint dir.
    This function returns a state_dict (CPU-mapped).
    """
    # Common cases:
    #  - PyTorch: ckpt_path is a file saved via torch.save(state_dict)
    #  - DeepSpeed stage 3: ckpt_path is a folder with 'mp_rank_00' etc or zero3 checkpoints (need conversion)
    #  - Some trainers saved 'state_dict.pt' inside the ckpt folder
    
    if os.path.isfile(ckpt_path):
        # try simple load
        sd = torch.load(ckpt_path, map_location='cpu')
        # if it wraps {'state_dict': ...}
        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            return sd['state_dict']
        return sd
    elif os.path.isdir(ckpt_path):
        # look for common filenames
        for cand in ['pytorch_model.bin', 'model.pt', 'model.pth', 'state_dict.pt', 'state_dict.pth']:
            fp = os.path.join(ckpt_path, cand)
            if os.path.isfile(fp):
                sd = torch.load(fp, map_location='cpu')
                if isinstance(sd, dict) and 'state_dict' in sd:
                    return sd['state_dict']
                return sd
        # fallback: try DeepSpeed conversion if requested and available
        if try_deepspeed:
            try:
                import deepspeed
                # deepspeed provides zero_to_fp32.py in its repo; but programmatic call is not always stable.
                # Use official deepspeed.utils to load zero checkpoints if available.
                from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
                # This API expects a model-like object to restore into; we instead ask for full state dict:
                sd = load_state_dict_from_zero_checkpoint(checkpoint_dir=ckpt_path)
                return sd
            except Exception:
                pass
        # last resort: load any tensor file in dir
        candidates = []
        for root, _, files in os.walk(ckpt_path):
            for f in files:
                if f.endswith(('.pt', '.pth', '.bin')):
                    candidates.append(os.path.join(root, f))
        if candidates:
            sd = torch.load(candidates[0], map_location='cpu')
            if isinstance(sd, dict) and 'state_dict' in sd:
                return sd['state_dict']
            return sd
    raise RuntimeError(f"Cannot load state_dict from {ckpt_path}")

def weighted_average_state_dicts(sd_list, weights):
    """
    sd_list: list of state_dicts (each a dict mapping name->tensor on cpu)
    weights: list of floats (same length)
    returns averaged state_dict
    """
    avg = {}
    total_w = float(sum(weights))
    # gather all keys
    keys = set()
    for sd in sd_list:
        keys.update(sd.keys())
    for k in keys:
        # initialize accumulator on CPU
        acc = None
        for sd, w in zip(sd_list, weights):
            if k not in sd:
                continue  # missing param - treat as zero contribution
            t = sd[k].cpu().to(torch.float64)  # do accumulation in double to reduce numeric error
            if acc is None:
                acc = w * t
            else:
                acc = acc + w * t
        if acc is None:
            continue
        avg[k] = (acc / total_w).to(torch.float32)
    return avg

def save_state_dict(sd, out_path):
    torch.save(sd, out_path)
    print(f"Saved aggregated state_dict to {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db-url', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--filter', default=None, help="SQL LIKE pattern passed to filter path, e.g. '%shard-prefix%'")
    p.add_argument('--finetune', action='store_true', help='Run a short finetune after aggregation')
    p.add_argument('--val-shards', default=None, help='Comma-separated list of validation shard paths or glob pattern')
    p.add_argument('--deepspeed-convert', action='store_true', help='Try to convert DeepSpeed checkpoints')
    args = p.parse_args()

    conn = psycopg2.connect(args.db_url)
    rows = fetch_checkpoints(conn, args.filter)
    if not rows:
        print("No checkpoints found with given filter.")
        return

    sd_list = []
    weights = []
    for (path, metrics, examples_processed) in rows:
        print("Loading", path)
        sd = load_state_dict_maybe_deepspeed(path, try_deepspeed=args.deepspeed_convert)
        sd_list.append(sd)
        w = examples_processed if examples_processed and examples_processed>0 else 1.0
        weights.append(float(w))

    print("Averaging", len(sd_list), "state_dicts")
    agg = weighted_average_state_dicts(sd_list, weights)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_state_dict(agg, args.out)

    # Optionally run a short finetune: delegate to adaptation_trainer (fast)
    if args.finetune:
        print("Launching short fine-tune (CPU init). Delegating to adaptation_trainer.py")
        cmd = f"python adaptation_trainer.py --db-url '{args.db_url}' --checkpoint '{args.out}' --steps 200 --lr 1e-5 --work-dir '{os.path.dirname(args.out)}'"
        os.system(cmd)

if __name__ == "__main__":
    main()
