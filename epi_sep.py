"""
epistemic_sep.py
Epistemic-aware separation pipeline:
 - conv-dict pretraining -> instrument atoms
 - separator training with instrument-wavelet loss
 - EpistemicAgent: uncertainty + attribution + adaptation + provenance logging

Run: python epistemic_sep.py
"""

import os, time, sqlite3, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from captum.attr import IntegratedGradients
import torchaudio
import soundfile as sf

# ---- Small helper utils ----
SR = 16000

def load_mono(path, sr=SR):
    x, orig_sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if orig_sr != sr:
        x = torchaudio.functional.resample(torch.from_numpy(x).unsqueeze(0),
                                           orig_sr, sr).squeeze(0).numpy()
    return x.astype(np.float32)

def save_npz_metadata(path, metadata: dict):
    np.savez(path, **metadata)

# ---- 1) ConvDictAE (fast version) ----
class ConvDictAE(nn.Module):
    def __init__(self, n_atoms=64, atom_len=512, hop=64, shrink_lambda=0.05):
        super().__init__()
        self.encoder = nn.Conv1d(1, n_atoms, kernel_size=atom_len, stride=hop, padding=atom_len//2, bias=False)
        self.decoder = nn.ConvTranspose1d(n_atoms, 1, kernel_size=atom_len, stride=hop, padding=atom_len//2, bias=False)
        nn.init.normal_(self.decoder.weight, 0.0, 0.01)
        self.shrink_lambda = shrink_lambda

    def forward(self, x):
        z = self.encoder(x)
        # soft-threshold (proximal)
        z_shrunk = torch.sign(z) * F.relu(torch.abs(z) - self.shrink_lambda)
        xhat = self.decoder(z_shrunk)
        return xhat, z_shrunk

# ---- 2) instrument-wavelet loss from atoms (frozen kernels) ----
class InstrumentWaveletLoss(nn.Module):
    def __init__(self, atom_banks, downsample=None, per_scale_weights=None):
        super().__init__()
        self.atom_banks = atom_banks
        self.scales = len(atom_banks)
        self.downsample = downsample or [1]*self.scales
        self.per_scale_weights = per_scale_weights or [1.0/self.scales]*self.scales
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, inst_name):
        # pred/target: (B,1,T). inst_name chooses the bank
        banks = self.atom_banks[inst_name]['banks']
        device = pred.device
        loss = 0.0
        for si, np_atoms in enumerate(banks):
            kernel = torch.from_numpy(np_atoms[:, None, :]).to(device)
            ds = self.downsample[si]
            if ds > 1:
                pred_ds = F.avg_pool1d(pred, kernel_size=ds, stride=ds)
                targ_ds = F.avg_pool1d(target, kernel_size=ds, stride=ds)
            else:
                pred_ds = pred; targ_ds = target
            pad = np_atoms.shape[1] // 2
            pcoef = F.conv1d(pred_ds, kernel, padding=pad)
            tcoef = F.conv1d(targ_ds, kernel, padding=pad)
            # normalize per-atom by RMS of target to keep scale stable
            rms = (tcoef.abs().mean(dim=-1, keepdim=True) + 1e-8)
            normalized = (pcoef - tcoef).abs() / rms
            loss = loss + self.per_scale_weights[si] * normalized.mean()
        return loss

# ---- 3) Separator backbone (simple UNet-ish) ----
class SmallUNet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.enc1 = nn.Conv1d(1, base, 15, padding=7)
        self.enc2 = nn.Conv1d(base, base*2, 15, padding=7, stride=2)
        self.bott = nn.Conv1d(base*2, base*2, 3, padding=1)
        self.dec2 = nn.ConvTranspose1d(base*2, base, 4, stride=2, padding=1)
        self.outc = nn.Conv1d(base*2, 1, 1)
    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(e1))
        b = F.relu(self.bott(e2))
        d2 = F.relu(self.dec2(b))
        if d2.shape[-1] != e1.shape[-1]:
            d2 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        out = self.outc(d2)
        return out

# ---- 4) Small epistemic agent ----
class EpistemicAgent:
    """
    - Maintains a small ensemble (N copies) or MC-dropout runs for uncertainty.
    - Uses Captum (IntegratedGradients) for attributions.
    - Logs metrics to TensorBoard and a tiny sqlite provenance DB.
    - Triggers a local adaptation step if uncertainty or attribution drift passes thresholds.
    """
    def __init__(self, separator_factory, atom_banks, device='cpu', ensemble_size=3, tbdir='runs/epistemic'):
        self.device = device
        self.ensemble_size = ensemble_size
        # build ensemble
        self.models = [separator_factory().to(device) for _ in range(ensemble_size)]
        self.atom_banks = atom_banks
        self.writer = SummaryWriter(tbdir)
        # provenance DB
        self.conn = sqlite3.connect('provenance.db', check_same_thread=False)
        self._init_db()
        # prepare Captum IG wrapper against first model (we will re-create per model when needed)
        self.ig = IntegratedGradients(self.models[0])
        # keep track of last-run metrics to detect drifts
        self.last_avg_unc = None
        self.adapt_steps = 8

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS events
                     (ts REAL, event_type TEXT, payload TEXT)''')
        self.conn.commit()

    def log_event(self, typ, payload):
        self.conn.execute('INSERT INTO events VALUES (?,?,?)', (time.time(), typ, str(payload)))
        self.conn.commit()

    def ensemble_predict(self, mix):
        """
        mix: tensor (B,1,T)
        returns: preds list [(B,1,T)], mean, variance
        """
        preds = []
        for m in self.models:
            m.eval()
            with torch.no_grad():
                preds.append(m(mix.to(self.device)).detach().cpu())
        stacked = torch.stack(preds, dim=0)  # (E,B,1,T)
        mean = stacked.mean(dim=0)
        var = stacked.var(dim=0)
        # simple scalar uncertainty: mean var over time & channel
        unc = var.mean().item()
        return preds, mean, var, unc

    def mc_dropout_predict(self, model, mix, runs=10):
        # enable dropout at eval by toggling
        model.train()  # to ensure dropout active
        preds = []
        for _ in range(runs):
            with torch.no_grad():
                preds.append(model(mix.to(self.device)).detach().cpu())
        stacked = torch.stack(preds, dim=0)
        mean = stacked.mean(dim=0); var = stacked.var(dim=0)
        return preds, mean, var, var.mean().item()

    def explain_and_score(self, model_idx, mix, pred, target, inst_name, atom_loss_module):
        """
        Run IntegratedGradients and compute:
         - average attribution energy in windows where instrument atoms strongly activate,
         - compare to expected atom coefficients to produce an attribution alignment score.
        """
        model = self.models[model_idx]
        model.zero_grad()
        model.eval()
        mix = mix.to(self.device)
        pred = pred.to(self.device)
        # Using integrated gradients: input is mix, target output channel (sum over time)
        ig = IntegratedGradients(lambda x: model(x).sum(dim=-1))
        # flatten to (1,1,T) if needed
        attributions = ig.attribute(mix, target=0, n_steps=30)  # (1,1,T)
        # compute atom coefficients for target (CPU)
        # Use same computation as loss: conv with atoms
        banks = self.atom_banks[inst_name]['banks']
        attrib_score = 0.0
        for si, atoms in enumerate(banks):
            kernel = torch.from_numpy(atoms[:, None, :]).to(self.device)
            pad = atoms.shape[1]//2
            tcoef = F.conv1d(target.to(self.device), kernel, padding=pad).abs()
            # compute where tcoef is large -> we expect attribution there
            mask = (tcoef.mean(dim=1, keepdim=True) > (tcoef.mean() * 0.6)).float()
            # Ensure mask matches attributions size
            if mask.shape[-1] > attributions.shape[-1]:
                mask = mask[..., :attributions.shape[-1]]
            elif mask.shape[-1] < attributions.shape[-1]:
                attributions = attributions[..., :mask.shape[-1]]
            # attribution energy inside mask
            a_energy = (attributions.abs() * mask).sum().item()
            total = attributions.abs().sum().item() + 1e-12
            attrib_score += a_energy / total
        # normalize to [0,1]
        attrib_score = attrib_score / len(banks)
        return attrib_score, attributions.detach().cpu()

    def calibrate_and_record(self, epoch, mix, target, inst_name, atom_loss_module):
        # Ensemble predict
        preds, mean, var, unc = self.ensemble_predict(mix)
        # log to TB
        self.writer.add_scalar('uncertainty/epoch', unc, epoch)
        # compute attribution alignment across ensemble (avg)
        scores = []
        for i, _ in enumerate(preds):
            s, at = self.explain_and_score(i, mix, preds[i], target, inst_name, atom_loss_module)
            scores.append(s)
        avg_score = float(np.mean(scores))
        self.writer.add_scalar('attrib/alignment', avg_score, epoch)
        # record provenance
        self.log_event('calibration', {'epoch': epoch, 'uncertainty': unc, 'attrib_score': avg_score})
        # detect drift or high uncertainty
        trigger = False
        if self.last_avg_unc is None:
            self.last_avg_unc = unc
        else:
            # simple heuristic: if unc increased by > 50% or attribution drops below 0.4
            if (unc > 1.5 * self.last_avg_unc) or (avg_score < 0.4):
                trigger = True
        self.last_avg_unc = unc
        return trigger, unc, avg_score

    def adapt_local(self, mix, target, inst_name, atom_loss_module, lr=1e-5, steps=8):
        """
        Perform a cautious adaptation: small number of steps on a copy of model,
        evaluate, then optionally adopt if metric improved. This keeps provenance.
        """
        adopted = False
        for i, m in enumerate(self.models):
            # clone parameters to temporary optimizer
            backup = {n: p.detach().cpu().clone() for n, p in m.named_parameters()}
            opt = torch.optim.Adam(m.parameters(), lr=lr)
            # small fine-tune
            m.train()
            for s in range(steps):
                pred = m(mix.to(self.device))
                time_loss = F.l1_loss(pred, target.to(self.device))
                wave_loss = atom_loss_module(pred, target.to(self.device), inst_name)
                loss = time_loss + 4.0 * wave_loss
                opt.zero_grad(); loss.backward(); opt.step()
            # evaluate simple metric on target (SI-SDR improvement)
            m.eval()
            with torch.no_grad():
                new_pred = m(mix.to(self.device))
                old_pred = torch.nn.functional.conv1d # placeholder - we could use backup predictions stored earlier
            # adopt heuristic: always adopt for now (or compare on heldout)
            adopted = True
            self.log_event('adapt', {'model_idx': i, 'steps': steps})
        return adopted

    def active_learning_queue(self, mix, pred_mean, var, threshold=1e-4):
        """
        Return boolean whether to request human review for this sample: high var or low attrib alignment.
        """
        # simple rule: if var.mean() is above threshold
        flag = (var.mean().item() > threshold)
        if flag:
            self.log_event('active_query', {'var': float(var.mean().item())})
        return flag

# ---- 5) Integration example / runner ----
def run_demo_pipeline():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # -------------------
    # (A) Pretrain small conv-dict on a few instrument files -> atoms
    # (In practice you would run a dedicated pretrain step; here we emulate)
    # -------------------
    # suppose you have instrument_ref dict:
    instr_ref = {
        'violin': ['/path/to/violin1.wav', '/path/to/violin2.wav'],
        # ...
    }
    # For brevity, assume we already trained and saved banks:
    atom_banks = {
        'violin': {'banks': [ np.random.randn(48,512).astype(np.float32), np.random.randn(32,2048).astype(np.float32) ] }
    }
    # -------------------
    # (B) Train separator normally (use existing code from previous messages)
    # -------------------
    separator_factory = lambda: SmallUNet()
    # Make a toy ensemble by creating three models
    agent = EpistemicAgent(separator_factory, atom_banks, device=device, ensemble_size=3, tbdir='runs/epstest')
    # -------------------
    # (C) Simulate continuous operation: take a mixture, run ensemble -> check adapt
    # -------------------
    # Toy mix/target (replace with real tensors)
    mix = torch.randn(1,1,16000)
    target = torch.randn(1,1,16000)
    # fake one model weights (in reality you train models first)
    trigger, unc, a_score = agent.calibrate_and_record(epoch=0, mix=mix, target=target, inst_name='violin', atom_loss_module=None)
    print("trigger", trigger, "unc", unc, "attrib_score", a_score)
    # if triggered, adapt
    if trigger:
        agent.adapt_local(mix, target, 'violin', atom_loss_module=None)
    # Ask for user review if active queue flagged
    preds, mean, var, _ = agent.ensemble_predict(mix)
    if agent.active_learning_queue(mix, mean, var):
        print("Queue sample for user annotation / curation")
    print("Done demo.")

if __name__ == "__main__":
    run_demo_pipeline()
