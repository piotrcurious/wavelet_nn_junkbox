#!/usr/bin/env python3
# epistemic_agent_orchestrator.py
# Usage:
#   python epistemic_agent_orchestrator.py --db-url postgresql://user:pass@db:5432/sepdb \
#        --val-shards '/shared/data/val/*.tar' --poll 60 --agent-id agent001

import argparse, time, glob, os, json
import numpy as np
import torch
import psycopg2
from psycopg2.extras import Json
from fractal_bayes import FractalBayesRefiner, dyadic_wavelet_coeffs_np  # assumes fractal_bayes.py reachable
import webdataset as wds

# --- helper: sample a few validation segments from shards (streamed) ---
def sample_validation_segments(val_glob, n=8, seglen=16000):
    shards = sorted(glob.glob(val_glob))
    segs = []
    for s in shards:
        ds = wds.WebDataset(s).decode(wds.audio.AudioFileHandler).to_tuple("wav")
        for i, (wav,) in enumerate(ds):
            if len(segs) >= n:
                break
            if wav.ndim>1:
                wav = wav.mean(axis=0)
            # trim/pad to seglen
            if len(wav) >= seglen:
                start = np.random.randint(0, len(wav)-seglen)
                seg = wav[start:start+seglen]
            else:
                seg = np.pad(wav, (0, seglen - len(wav)))
            segs.append(seg.astype(np.float32))
        if len(segs) >= n:
            break
    return segs

# --- helper: load model state_dict on CPU and build inference model using user-specified factory ---
def load_model_from_state(state_path, model_factory):
    # state_path is file path to state_dict (pt)
    sd = torch.load(state_path, map_location='cpu')
    if 'state_dict' in sd:
        sd = sd['state_dict']
    model = model_factory()
    # load to CPU then move to eval device (use CPU here)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model

# --- compute ensemble predictions (on CPU, small sample) ---
def ensemble_predict_cpu(models, segment_torch):
    # segment_torch: torch.tensor (1,1,T) on CPU
    preds = []
    for m in models:
        with torch.no_grad():
            preds.append(m(segment_torch).cpu().numpy())
    stacked = np.stack(preds, axis=0) # (E,1,1,T) shapes vary
    mean = stacked.mean(axis=0)
    var = stacked.var(axis=0)
    return preds, mean, var

# --- orchestrator main loop ---
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--db-url', required=True)
    p.add_argument('--val-shards', required=True)
    p.add_argument('--poll', type=int, default=60)
    p.add_argument('--agent-id', default='agent01')
    p.add_argument('--model-factory', default=None, help='python import path to a factory function returning model() for inference')
    p.add_argument('--atom-banks-npy', default=None, help='path to directory with instrument atom npy banks (keyed by instrument)')
    args = p.parse_args()

    # load atom banks dict if provided: expects files like violin_atoms_scale0.npy etc.
    atom_banks = {}
    if args.atom_banks_npy:
        # naive loader: groups by prefix before _atoms_
        for f in glob.glob(os.path.join(args.atom_banks_npy, '*.npy')):
            bn = os.path.basename(f)
            # expect format: <inst>_atoms_scale<idx>.npy
            parts = bn.split('_atoms_')
            if len(parts) != 2:
                continue
            inst = parts[0]
            scale_part = parts[1]  # scaleX.npy
            idx = int(scale_part.replace('scale','').split('.')[0])
            atom = np.load(f)
            atom_banks.setdefault(inst, {'banks':[]})['banks'].append(atom)
        # ensure order by scale index
        for inst in atom_banks:
            atom_banks[inst]['banks'] = sorted(atom_banks[inst]['banks'], key=lambda x: x.shape[1]) # best-effort

    # instantiate fractal refiner
    refiner = FractalBayesRefiner(atom_banks, wavelet='db4', max_level=6, device='cpu')

    # model factory import if provided
    model_factory = None
    if args.model_factory:
        # for example: mymodule:make_model
        mod, fn = args.model_factory.split(':')
        m = __import__(mod, fromlist=[fn])
        model_factory = getattr(m, fn)
    else:
        # fallback simple CPU model factory to match your separator
        from types import SimpleNamespace
        def dummy_model():
            # user should provide an actual model factory for inference
            net = SimpleNamespace()
            def forward(x):
                return x
            net.__call__ = forward
            return net
        model_factory = dummy_model

    conn = psycopg2.connect(args.db_url)

    # sample validation segments once up-front (small)
    val_segments = sample_validation_segments(args.val_shards, n=8, seglen=16000)
    print("Loaded", len(val_segments), "validation segments")

    while True:
        # fetch recent checkpoints (limit last N)
        cur = conn.cursor()
        cur.execute("SELECT ckpt_id, path, metrics FROM checkpoints ORDER BY ts DESC LIMIT 20")
        rows = cur.fetchall()
        cur.close()

        for ckpt_id, ckpt_path, metrics in rows:
            # skip if we've already evaluated this checkpoint (we store events log with type 'agent_eval')
            ccur = conn.cursor()
            ccur.execute("SELECT 1 FROM events WHERE event_type='agent_eval' AND payload->>'ckpt' = %s", (ckpt_path,))
            if ccur.fetchone():
                ccur.close()
                continue
            ccur.close()

            # load model(s) - for ensemble we may use multiple checkpoints around this ckpt or use same model multiple times with dropout.
            # For simplicity, we load one model (or up to 3 nearby) if present
            candidate_paths = [ckpt_path]
            # try to find some sibling checkpoints (same directory)
            base_dir = os.path.dirname(ckpt_path)
            sibling = glob.glob(os.path.join(base_dir, '*.pt'))[:2]
            for s in sibling:
                if s not in candidate_paths:
                    candidate_paths.append(s)
            # load models to CPU using factory
            models = []
            for pth in candidate_paths:
                try:
                    model = load_model_from_state(pth, model_factory)
                    models.append(model)
                except Exception as e:
                    print("Failed to load model", pth, e)
            if not models:
                print("No models loaded for", ckpt_path, "skipping")
                # log event
                cur = conn.cursor()
                cur.execute("INSERT INTO events (event_type, payload) VALUES (%s, %s)", ('agent_eval', Json({'ckpt': ckpt_path, 'status': 'load_failed'})))
                conn.commit(); cur.close()
                continue

            # evaluate models on validation segments: compute ensemble variance and fractal diagnostics
            ensemble_vars = []
            slope_list = []
            shrink_list = []
            for seg in val_segments:
                seg_t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0)  # (1,1,T)
                preds, mean, var = ensemble_predict_cpu(models, seg_t)
                ensemble_vars.append(float(np.mean(var)))
                # compute fractal diagnostics using refiner: refined vs target (we don't have real target here -> use mean as proxy)
                # best case: you supply paired (mix,target) validation pairs; here we use mean as a rough stand-in
                pred_np = mean.squeeze()
                # for demonstration we provide target == pred (so shrinkage ~ small) - in practice use real target
                refined, aux_loss, diag = refiner.refine_prediction(pred_np, pred_np, inst_name=list(atom_banks.keys())[0] if atom_banks else 'unknown', atoms_scale_list=atom_banks[list(atom_banks.keys())[0]]['banks'] if atom_banks else [])
                slope_list.append(diag.get('slope', 0.0))
                shrink_list.append(diag.get('mean_shrinkage', 0.0))

            avg_unc = float(np.mean(ensemble_vars))
            avg_slope = float(np.mean(slope_list)) if slope_list else 0.0
            avg_shrink = float(np.mean(shrink_list)) if shrink_list else 0.0

            # log agent evaluation event
            cur = conn.cursor()
            cur.execute("INSERT INTO events (event_type, payload) VALUES (%s, %s)", ('agent_eval', Json({'ckpt': ckpt_path, 'unc': avg_unc, 'slope': avg_slope, 'shrink': avg_shrink})))
            conn.commit(); cur.close()

            # decision rule: if uncertainty high AND mean_shrink large -> schedule adaptation
            UNC_TH = 1e-4   # tune thresholds for your data
            SHRINK_TH = 0.2
            if (avg_unc > UNC_TH and avg_shrink > SHRINK_TH):
                # create adaptation job
                cur = conn.cursor()
                params = {'lr': 1e-5, 'steps': 800, 'micro_batch': 2, 'val_shards': args.val_shards}
                cur.execute("INSERT INTO adaptation_jobs (checkpoint_path, requested_by, status, params) VALUES (%s,%s,%s,%s) RETURNING adapt_id", (ckpt_path, args.agent_id, 'pending', Json(params)))
                adapt_id = cur.fetchone()[0]
                conn.commit(); cur.close()
                print("Scheduled adaptation job", adapt_id, "for", ckpt_path)
            else:
                print("No adaptation needed for", ckpt_path, "unc", avg_unc, "shrink", avg_shrink)

        # sleep until next poll
        time.sleep(args.poll)

if __name__ == "__main__":
    main()
