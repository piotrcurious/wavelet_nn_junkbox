#!/usr/bin/env python3
# adaptation_trainer.py
# Usage:
#  python adaptation_trainer.py --db-url postgresql://user:pass@db:5432/sepdb --adapt-id 123 --work-dir /shared/checkpoints

import argparse, os, time, json
import torch, torch.nn.functional as F
import psycopg2
from psycopg2.extras import Json
import glob
import webdataset as wds

# import your model factory
from my_model_module import make_model   # adapt to your repo

def claim_adaptation_job(conn, adapt_id, node_id):
    cur = conn.cursor()
    cur.execute("UPDATE adaptation_jobs SET status='running', claimed_by=%s, last_update=now() WHERE adapt_id=%s AND status='pending' RETURNING checkpoint_path, params", (node_id, adapt_id))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    if not row:
        raise RuntimeError("No pending adaptation job with id %s" % adapt_id)
    return row[0], row[1]

def mark_adapt_done(conn, adapt_id, node_id, ckpt_path, metrics):
    cur = conn.cursor()
    cur.execute("UPDATE adaptation_jobs SET status='completed', last_update=now(), claimed_by=%s WHERE adapt_id=%s", (node_id, adapt_id))
    cur.execute("INSERT INTO checkpoints (shard_id, node_id, path, metrics) VALUES (NULL, %s, %s, %s)", (node_id, ckpt_path, Json(metrics)))
    conn.commit()

def mark_adapt_failed(conn, adapt_id, node_id, reason):
    cur = conn.cursor()
    cur.execute("UPDATE adaptation_jobs SET status='failed', last_update=now(), claimed_by=%s WHERE adapt_id=%s", (node_id, adapt_id))
    cur.execute("INSERT INTO events (event_type, payload) VALUES (%s, %s)", ('adapt_failed', Json({'adapt_id': adapt_id, 'reason': reason})))
    conn.commit()

def load_checkpoint_to_model(checkpoint_path, model):
    sd = torch.load(checkpoint_path, map_location='cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=False)

def finetune(model, val_shards, lr=1e-5, steps=800, batch_size=2, device='cuda'):
    # Simple finetune on streamed validation shards (or small adaptation dataset)
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    gen = []
    # stream small amount of examples from val_shards
    shards = glob.glob(val_shards)
    for s in shards:
        ds = wds.WebDataset(s).decode(wds.audio.AudioFileHandler).to_tuple("wav")
        for (wav,) in ds:
            if wav.ndim>1:
                wav = wav.mean(axis=0)
            if len(wav) < 16000:
                wav = np.pad(wav, (0, 16000 - len(wav)))
            x = torch.from_numpy(wav.astype('float32')).unsqueeze(0).unsqueeze(0)
            gen.append(x)
            if len(gen) >= batch_size:
                break
        if len(gen) >= batch_size:
            break
    if not gen:
        raise RuntimeError("No validation samples found for adaptation")

    batch = torch.cat(gen, dim=0).to(device)
    for step in range(steps):
        opt.zero_grad()
        pred = model(batch)
        loss = F.l1_loss(pred, batch)   # use appropriate loss for your task
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print("adapt step", step, "loss", float(loss.item()))
    # return metrics
    return {'final_loss': float(loss.item()), 'steps': steps}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--db-url', required=True)
    parser.add_argument('--adapt-id', type=int, required=True)
    parser.add_argument('--node-id', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    conn = psycopg2.connect(args.db_url)
    try:
        ckpt_path, params = claim_adaptation_job(conn, args.adapt_id, args.node_id)
        print("Claimed adapt job", args.adapt_id, "checkpoint", ckpt_path, "params", params)
        model = make_model()
        load_checkpoint_to_model(ckpt_path, model)
        metrics = finetune(model, params['val_shards'], lr=params.get('lr',1e-5), steps=params.get('steps',200), device=args.device)
        # save adapted model
        outdir = os.path.join(args.work_dir, f"adapt_{args.adapt_id}")
        os.makedirs(outdir, exist_ok=True)
        outpath = os.path.join(outdir, "adapted_model.pt")
        torch.save({'state_dict': model.state_dict(), 'adapt_metrics': metrics}, outpath)
        mark_adapt_done(conn, args.adapt_id, args.node_id, outpath, metrics)
        print("Adaptation complete:", outpath)
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_adapt_failed(conn, args.adapt_id, args.node_id, str(e))
    finally:
        conn.close()
