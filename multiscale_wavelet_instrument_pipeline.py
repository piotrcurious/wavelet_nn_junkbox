"""
multiscale_instrument_wavelet_pipeline.py

Run on Linux with: python multiscale_instrument_wavelet_pipeline.py
Replace path placeholders with your files.

This script contains:
 - ConvDictAE: convolutional dictionary autoencoder (per-instrument).
 - Dataset & augmentations (pitch shift, reverb, distortion, harmonic intermods).
 - InstrumentWaveletLoss built from learned decoder kernels (multi-scale).
 - EnhancedSeparator with GainNorm, AdaptiveEQ, DeReverbNet + UNet backbone.
 - Pretrain dictionaries -> extract atoms -> train separator -> user-feedback fine-tune.
"""

import os, math, random, glob
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy import signal

# ---------------------------
# Utilities and augmentations
# ---------------------------
SR = 16000

def load_mono(path, sr=SR):
    x, orig_sr = sf.read(path, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if orig_sr != sr:
        x = librosa.resample(x.astype(np.float32), orig_sr, sr)
    return x.astype(np.float32)

def write_wav(path, x, sr=SR):
    sf.write(path, x.astype(np.float32), sr)

def random_pitch_shift(x, sr=SR, semitones_range=(-4,4)):
    n = random.uniform(*semitones_range)
    return librosa.effects.pitch_shift(x, sr=sr, n_steps=n)

def random_reverb(x, sr=SR, reverb_prob=0.8):
    if random.random() > reverb_prob:
        return x
    # simple Schroeder-like reverb via convolution with generated decaying noise
    rt60 = random.uniform(0.08, 0.6)
    len_imp = int(min(sr * rt60 * 2.0, sr*2))
    noise = np.random.randn(len_imp).astype(np.float32)
    # exponential decay
    times = np.arange(len_imp) / sr
    decay = np.exp(-times * 6.91/rt60)  # approximate
    imp = noise * decay
    imp = imp / (np.linalg.norm(imp) + 1e-12)
    wet = signal.fftconvolve(x, imp, mode='full')[:len(x)]
    mix = random.uniform(0.1, 0.6)
    return (1.0 - mix)*x + mix*wet

def random_distortion(x, gain_db_range=(0, 12), clip_prob=0.5):
    g = 10 ** (random.uniform(*gain_db_range) / 20.0)
    y = x * g
    if random.random() < clip_prob:
        thr = random.uniform(0.2, 0.9)
        y = np.clip(y, -thr, thr)
    # softclip
    y = np.tanh(y)
    # small lowpass to simulate intermod
    b, a = signal.butter(2, 8000/SR)
    y = signal.lfilter(b, a, y)
    return y

def intermodulation_mix(x, sr=SR):
    # add harmonics / intermod artifacts: sum of pitched & phase-shifted copies
    out = x.copy()
    for h in [2,3]:
        if random.random() < 0.5:
            pitch = random.choice([+7, -5, +12, -12]) / float(h)
            copy = librosa.effects.pitch_shift(x, sr=sr, n_steps=pitch)
            # small delay and attenuation
            att = random.uniform(0.02, 0.2)
            delay = int(random.uniform(0, 0.02)*sr)
            if delay > 0:
                copy = np.concatenate([np.zeros(delay, dtype=copy.dtype), copy])[:len(x)]
            out += att * copy
    # normalize
    out = out / (np.max(np.abs(out)) + 1e-9)
    return out

def augment_example(x, sr=SR):
    x = random_pitch_shift(x, sr) if random.random()<0.6 else x
    x = random_reverb(x, sr) if random.random()<0.9 else x
    x = random_distortion(x) if random.random()<0.7 else x
    x = intermodulation_mix(x) if random.random()<0.5 else x
    # final normalization
    x = x / (np.max(np.abs(x)) + 1e-9)
    return x

# ---------------------------
# ConvDict Autoencoder (per instrument)
# ---------------------------
class ConvDictAE(nn.Module):
    """
    Convolutional dictionary autoencoder:
     - Encoder: conv1d with out_channels = n_atoms producing coefficients c[t,atom]
       followed by non-linearity / soft-thresholding to encourage sparsity.
     - Decoder: conv_transpose1d using decoder kernels (atoms) to reconstruct.
    The decoder kernels are the learned atoms we will extract later.
    """
    def __init__(self, n_atoms=64, atom_len=512, hop=64, shrink_lambda=0.1):
        super().__init__()
        self.n_atoms = n_atoms
        self.atom_len = atom_len
        self.hop = hop
        stride = hop
        # encoder conv produces coefficients (with overlap depending on stride)
        self.encoder = nn.Conv1d(1, n_atoms, kernel_size=atom_len, stride=stride, padding=atom_len//2, bias=False)
        # decoder convtranspose reconstructs from coefficients
        # We want decoder kernels shaped (n_atoms, 1, atom_len)
        self.decoder = nn.ConvTranspose1d(n_atoms, 1, kernel_size=atom_len, stride=stride, padding=atom_len//2, bias=False)
        # initialize decoder with small random atoms (better to init with random windows)
        nn.init.normal_(self.decoder.weight, mean=0.0, std=0.01)
        # optionally tie weights: decoder weights are the atoms we want; encoder could be transposed later if desired
        self.shrink_lambda = shrink_lambda

    def forward(self, x):
        # x: (B,1,T)
        coeff = self.encoder(x)  # (B, n_atoms, T')
        # soft shrinkage (proximal operator for L1)
        thresh = self.shrink_lambda
        coeff_shrunk = torch.sign(coeff) * F.relu(torch.abs(coeff) - thresh)
        recon = self.decoder(coeff_shrunk)  # (B,1,T)
        return recon, coeff_shrunk

# ---------------------------
# Instrument dataset that auto-encodes & augments to produce training set
# ---------------------------
class InstrumentRefDataset(torch.utils.data.Dataset):
    """
    Given a list of reference files for one instrument, this dataset returns random short segments,
    optionally augmented, for training the dictionary autoencoder.
    """
    def __init__(self, file_list, sr=SR, seg_len=4096, augment=True):
        self.files = list(file_list)
        self.sr = sr
        self.seg_len = seg_len
        self.augment = augment
        # pre-load durations? for simplicity keep paths and load on the fly

    def __len__(self):
        return len(self.files) * 100  # heuristics

    def __getitem__(self, idx):
        path = random.choice(self.files)
        x = load_mono(path, sr=self.sr)
        if len(x) <= self.seg_len:
            # pad
            x = np.pad(x, (0, self.seg_len - len(x)))
            segment = x
        else:
            start = random.randint(0, len(x) - self.seg_len)
            segment = x[start:start+self.seg_len]
        if self.augment:
            segment = augment_example(segment, sr=self.sr)
        # normalize
        segment = segment / (np.max(np.abs(segment)) + 1e-9)
        return torch.from_numpy(segment).unsqueeze(0).float()  # (1, T)

# ---------------------------
# Train ConvDictAE for instrument -> obtain atoms
# ---------------------------
def train_conv_dict_ae(ref_files, n_atoms=64, atom_len=512, hop=64, epochs=8, batch_size=8, lr=1e-3, device='cuda'):
    ds = InstrumentRefDataset(ref_files, seg_len=4096, augment=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    model = ConvDictAE(n_atoms=n_atoms, atom_len=atom_len, hop=hop).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total_loss = 0.0
        for i, batch in enumerate(loader):
            x = batch.to(device)  # (B,1,T)
            if x.shape[-1] < atom_len:
                continue
            recon, coeff = model(x)
            # loss = L1 recon + small L1 on coefficients for sparsity
            recon_loss = F.l1_loss(recon, x)
            sparsity = 1e-3 * torch.mean(torch.abs(coeff))
            loss = recon_loss + sparsity
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if i % 100 == 0 and i>0:
                print(f"ep{ep} it{i} avg-loss:{total_loss/(i+1):.4f}")
            # limit iterations per epoch for speed during experimentation
            if i > 400:
                break
        print(f"Epoch {ep} avg loss {total_loss/(i+1):.6f}")
    # extract decoder kernels as numpy atoms
    atoms = model.decoder.weight.detach().cpu().numpy()  # shape (1_outch? careful)
    # decoder.weight shape: (out_channels=1, in_channels=n_atoms, kernel)???
    # In torch ConvTranspose1d, weight shape is (in_channels, out_channels, k)
    # Our decoder was created with (n_atoms -> 1), so weight shape is (n_atoms, 1, k)
    atoms = atoms[:, 0, :]  # (n_atoms, atom_len)
    # normalize each atom
    atoms = atoms / (np.linalg.norm(atoms, axis=1, keepdims=True) + 1e-12)
    return atoms, model

# ---------------------------
# Multi-scale atom bank builder
# ---------------------------
def build_multiscale_banks(instrument_ref_dirs, scales=[(512,64),(2048,256)], n_atoms_per_scale=64, device='cuda'):
    # instrument_ref_dirs: dict{name: [file_paths]}
    instrument_banks = {}
    # For each instrument: train conv-dict per scale
    for inst, files in instrument_ref_dirs.items():
        print(f"Building atom banks for instrument {inst} ...")
        banks = []
        models = []
        for (atom_len, hop) in scales:
            atoms, model = train_conv_dict_ae(files, n_atoms=n_atoms_per_scale, atom_len=atom_len, hop=hop, epochs=4, batch_size=6, device=device)
            banks.append(atoms)
            models.append(model)
        instrument_banks[inst] = {'banks': banks, 'models': models}
    return instrument_banks

# ---------------------------
# InstrumentWaveletLoss (multi-scale learned atoms)
# ---------------------------
class InstrumentWaveletLoss(nn.Module):
    def __init__(self, instrument_banks, downsample_factors=None, per_scale_weights=None, device='cuda'):
        """
        instrument_banks: dict { inst_name: list of np arrays atoms per scale }
        We will compute loss w.r.t the *target* instrument(s) present in training pair.
        If training single-instrument task, pass keys accordingly.
        downsample_factors: list per scale
        """
        super().__init__()
        self.inst_banks = instrument_banks
        # store kernels as buffers per inst & scale
        self.scales = len(next(iter(instrument_banks.values()))['banks'])
        self.downsample_factors = downsample_factors or [1]*self.scales
        # weights per scale
        if per_scale_weights is None:
            self.per_scale_weights = [1.0/ self.scales] * self.scales
        else:
            self.per_scale_weights = per_scale_weights
        # We'll not tie atoms to class inside the module; at forward we will pick atoms according to instrument id list passed
        self.l1 = nn.L1Loss(reduction='mean')

    def forward(self, pred, target, inst_name):
        """
        pred, target: (B,1,T)
        inst_name: string, which instrument's bank to use (assume all batch items are same instrument)
        """
        device = pred.device
        bankset = self.inst_banks[inst_name]['banks']
        loss = 0.0
        for si, atoms in enumerate(bankset):
            # atoms is numpy (n_atoms, L)
            kernel = torch.from_numpy(atoms[:, None, :]).to(device)  # (n_atoms,1,L)
            ds = self.downsample_factors[si]
            if ds > 1:
                pred_ds = F.avg_pool1d(pred, kernel_size=ds, stride=ds, padding=0)
                targ_ds = F.avg_pool1d(target, kernel_size=ds, stride=ds, padding=0)
            else:
                pred_ds = pred; targ_ds = target
            L = atoms.shape[1]
            pad = L // 2
            pcoef = F.conv1d(pred_ds, kernel, padding=pad)  # (B, n_atoms, T_ds)
            tcoef = F.conv1d(targ_ds, kernel, padding=pad)
            w = self.per_scale_weights[si]
            # optionally normalize per-atom by target RMS to prevent low-energy atoms dominating
            atom_rms = (tcoef.abs().mean(dim=-1, keepdim=True) + 1e-8)  # (B,n_atoms,1)
            normalized_diff = torch.abs(pcoef - tcoef) / atom_rms
            # mean over atoms/time/batch
            scale_loss = w * normalized_diff.mean()
            loss = loss + scale_loss
        return loss

# ---------------------------
# Enhanced Separator (gainnorm, eq, dereverb, backbone)
# ---------------------------
class GainNorm(nn.Module):
    def __init__(self, pool_kernel=256):
        super().__init__()
        self.pool = nn.AvgPool1d(pool_kernel, stride=pool_kernel//2, padding=pool_kernel//4)
        self.fc = nn.Sequential(
            nn.Conv1d(1, 16, 1), nn.ReLU(),
            nn.Conv1d(16, 1, 1), nn.Sigmoid()
        )
    def forward(self, x):
        # x (B,1,T)
        s = torch.log1p(torch.sqrt(self.pool(x**2).clamp(min=1e-9)))
        g = self.fc(s)
        g_up = F.interpolate(g, size=x.shape[-1], mode='linear', align_corners=False)
        return x * (1.0 / (1e-3 + g_up))

class AdaptiveEQ(nn.Module):
    def __init__(self, n_filters=12, kernel=65):
        super().__init__()
        self.nf = n_filters
        self.filters = nn.Parameter(torch.randn(n_filters, 1, kernel) * 0.01)
        self.attn = nn.Sequential(nn.Conv1d(1, n_filters, 1), nn.Softmax(dim=1))
    def forward(self, x):
        w = self.attn(x)
        convs = []
        for i in range(self.nf):
            f = self.filters[i:i+1,:,:]
            convs.append(F.conv1d(x, f, padding=f.shape[-1]//2))
        convs = torch.cat(convs, dim=1)
        out = (w * convs).sum(dim=1, keepdim=True)
        return out

class DeReverbNet(nn.Module):
    def __init__(self, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, base, 31, padding=15),
            nn.ReLU(),
            nn.Conv1d(base, base, 31, padding=30, dilation=2),
            nn.ReLU(),
            nn.Conv1d(base, 1, 31, padding=15),
        )
    def forward(self, x):
        pred_rev = self.net(x)
        return x - pred_rev

class SeparatorUNet(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.enc1 = nn.Conv1d(in_ch, base, 15, padding=7)
        self.enc2 = nn.Conv1d(base, base*2, 15, padding=7, stride=2)
        self.enc3 = nn.Conv1d(base*2, base*4, 15, padding=7, stride=2)
        self.bott = nn.Conv1d(base*4, base*4, 3, padding=1)
        self.dec3 = nn.ConvTranspose1d(base*4, base*2, 4, stride=2, padding=1)
        self.dec2 = nn.ConvTranspose1d(base*4, base, 4, stride=2, padding=1)
        self.outc = nn.Conv1d(base*2, 1, 1)
        self.act = nn.ReLU()
    def forward(self, x):
        e1 = self.act(self.enc1(x))
        e2 = self.act(self.enc2(e1))
        e3 = self.act(self.enc3(e2))
        b = self.act(self.bott(e3))
        d3 = self.act(self.dec3(b))
        # Ensure d3 and e2 have same size
        if d3.shape[-1] != e2.shape[-1]:
            d3 = F.interpolate(d3, size=e2.shape[-1], mode='linear', align_corners=False)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.act(self.dec2(d3))
        # Ensure d2 and e1 have same size
        if d2.shape[-1] != e1.shape[-1]:
            d2 = F.interpolate(d2, size=e1.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e1], dim=1)
        out = self.outc(d2)
        return out

class EnhancedSeparator(nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = GainNorm()
        self.eq = AdaptiveEQ()
        self.derev = DeReverbNet()
        self.unet = SeparatorUNet()
    def forward(self, mix):
        x = self.gain(mix)
        x = self.eq(x)
        x = self.derev(x)
        out = self.unet(x)
        # residual
        return out + x

# ---------------------------
# Training loop for separator with instrument wavelet loss + time loss + si-sdr optionally
# ---------------------------
def si_sdr_loss(est, ref, eps=1e-8):
    # expects (B,1,T)
    est = est.squeeze(1)
    ref = ref.squeeze(1)
    s_target = (torch.sum(est * ref, dim=1, keepdim=True) * ref) / (torch.sum(ref**2, dim=1, keepdim=True) + eps)
    e_noise = est - s_target
    si_sdr = 10 * torch.log10((torch.sum(s_target**2, dim=1) + eps) / (torch.sum(e_noise**2, dim=1) + eps) + eps)
    return -torch.mean(si_sdr)  # negative because we minimize

def train_separator(separator, inst_loss_module, dataloader, optimizer, device='cuda', epochs=10, lambda_wave=4.0, lambda_time=1.0):
    separator.train()
    for ep in range(epochs):
        tot = 0.0
        for i, (mix, target, inst_name) in enumerate(dataloader):
            mix = mix.to(device); target = target.to(device)
            if mix.shape[-1] < 128: # Avoid too small segments
                continue
            pred = separator(mix)
            time_loss = F.l1_loss(pred, target)
            wave_loss = inst_loss_module(pred, target, inst_name[0])  # assume batch same instrument for simplicity
            sdr_loss = si_sdr_loss(pred, target)
            loss = lambda_time * time_loss + lambda_wave * wave_loss + 0.1 * sdr_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tot += loss.item()
            if i % 20 == 0:
                print(f"ep{ep} it{i} loss {loss.item():.4f} time:{time_loss.item():.4f} wave:{wave_loss.item():.4f} sdr:{sdr_loss.item():.4f}")
            if i>400:
                break
        print(f"Epoch {ep} avg loss {tot/(i+1):.6f}")

# ---------------------------
# Simple dataloader that mixes instrument with background & returns instrument name
# ---------------------------
class MixtureDataset(torch.utils.data.Dataset):
    """
    Takes: dict instrument->{filelist}, background filelist
    Produces: (mix, instrument_target, instrument_name)
    The mix is instrument + background (and optional other instruments)
    """
    def __init__(self, instrument_files_dict, bg_files, sr=SR, seg_len=16000, snr_range=(0,6), augment=True):
        self.inst_dict = instrument_files_dict
        self.insts = list(instrument_files_dict.keys())
        self.bg_files = list(bg_files)
        self.sr = sr
        self.seg_len = seg_len
        self.snr_range = snr_range
        self.augment = augment
        # flatten file lists for quick sampling
        self.flat = []
        for k, v in instrument_files_dict.items():
            self.flat += [(k, p) for p in v]

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        inst, path = random.choice(self.flat)
        inst_sig = load_mono(path, sr=self.sr)
        if len(inst_sig) <= self.seg_len:
            inst_sig = np.pad(inst_sig, (0, self.seg_len - len(inst_sig)))
        else:
            st = random.randint(0, len(inst_sig)-self.seg_len)
            inst_sig = inst_sig[st:st+self.seg_len]
        if self.augment:
            inst_sig = augment_example(inst_sig, sr=self.sr)
        # sample background and mix at random SNR
        bg_path = random.choice(self.bg_files)
        bg = load_mono(bg_path, sr=self.sr)
        if len(bg) <= self.seg_len:
            bg = np.pad(bg, (0, self.seg_len - len(bg)))
        else:
            st = random.randint(0, len(bg)-self.seg_len)
            bg = bg[st:st+self.seg_len]
        snr = random.uniform(*self.snr_range)
        # scale instrument to desired SNR relative to background (RMS)
        rms_inst = np.sqrt(np.mean(inst_sig**2) + 1e-12)
        rms_bg = np.sqrt(np.mean(bg**2) + 1e-12)
        scale = (rms_bg * (10 ** (snr/20.0))) / (rms_inst + 1e-12)
        inst_scaled = inst_sig * scale
        mix = inst_scaled + bg
        # normalize mix to prevent clipping
        mix = mix / (np.max(np.abs(mix)) + 1e-9)
        inst_scaled = inst_scaled / (np.max(np.abs(inst_scaled)) + 1e-9)
        # return tensors
        mix_t = torch.from_numpy(mix).unsqueeze(0).float()
        inst_t = torch.from_numpy(inst_scaled).unsqueeze(0).float()
        return mix_t, inst_t, inst

# ---------------------------
# User feedback fine-tune helper
# ---------------------------
def user_feedback_finetune(separator, inst_loss_module, accepted_examples, optimizer, device='cuda', steps=200):
    """
    accepted_examples: list of tuples (mix_np, target_np, inst_name)
    We perform small number of gradient steps to fine-tune the separator on user-validated examples.
    """
    separator.train()
    for step in range(steps):
        mix_np, target_np, inst_name = random.choice(accepted_examples)
        mix = torch.from_numpy(mix_np).unsqueeze(0).unsqueeze(0).to(device)
        target = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0).to(device)
        pred = separator(mix)
        time_loss = F.l1_loss(pred, target)
        wave_loss = inst_loss_module(pred, target, inst_name)
        loss = 1.0 * time_loss + 4.0 * wave_loss
        optimizer.zero_grad()
        loss.backward()
        # gradient clipping small
        torch.nn.utils.clip_grad_norm_(separator.parameters(), 1.0)
        optimizer.step()
        if step % 50 == 0:
            print(f"fine-tune step {step} loss {loss.item():.4f}")

# ---------------------------
# Putting it all together (toy run)
# ---------------------------
def demo_pipeline(instrument_ref_dirs=None, bg_files=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Use provided paths or fall back to defaults
    if instrument_ref_dirs is None:
        instrument_ref_dirs = {
            'violin': glob.glob('data/violin_ref/*.wav'),
            'flute' : glob.glob('data/flute_ref/*.wav'),
        }
    if bg_files is None:
        bg_files = glob.glob('data/background_scenes/*.wav')
    # 1) Build multiscale banks (training conv-dict per instrument per scale)
    # Scales: (atom_len, hop)
    scales = [(512, 64), (2048, 256)]
    inst_banks = build_multiscale_banks(instrument_ref_dirs, scales=scales, n_atoms_per_scale=8, device=device)
    # 2) Prepare instrument-wavelet loss wrapper
    inst_loss_module = InstrumentWaveletLoss(inst_banks, downsample_factors=[1,4], per_scale_weights=[0.6, 0.4], device=device)
    inst_loss_module.to(device)
    # 3) Train separator
    mixture_dataset = MixtureDataset(instrument_ref_dirs, bg_files, seg_len=4096, augment=True)
    loader = torch.utils.data.DataLoader(mixture_dataset, batch_size=4, shuffle=True, num_workers=2)
    separator = EnhancedSeparator().to(device)
    opt = torch.optim.Adam(separator.parameters(), lr=1e-4)
    train_separator(separator, inst_loss_module, loader, opt, device=device, epochs=1)
    # 4) Simulate user feedback and fine-tune
    # create accepted_examples by running separator on some mix samples and optionally letting user mark them
    accepted = []
    for i in range(2):
        mix_t, inst_t, inst_name = mixture_dataset[i]
        mix_np = mix_t.squeeze(0).numpy()
        inst_np = inst_t.squeeze(0).numpy()
        accepted.append((mix_np, inst_np, inst_name))
    # fine-tune with a low lr
    opt2 = torch.optim.Adam(separator.parameters(), lr=1e-5)
    user_feedback_finetune(separator, inst_loss_module, accepted, opt2, device=device, steps=10)
    # Save models and atoms
    torch.save(separator.state_dict(), "separator_final.pth")
    # Save atom banks
    for inst, v in inst_banks.items():
        for si, atoms in enumerate(v['banks']):
            np.save(f"{inst}_atoms_scale{si}.npy", atoms)
    print("Demo pipeline finished.")

if __name__ == "__main__":
    demo_pipeline()
