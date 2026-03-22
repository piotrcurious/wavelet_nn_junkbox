import pytest
import torch
import numpy as np
import glob
from multiscale_wavelet_instrument_pipeline import build_multiscale_banks, InstrumentWaveletLoss, MixtureDataset, EnhancedSeparator, train_separator
import os

def test_full_pipeline_flow():
    device = 'cpu'

    # Use the mock data generated
    instrument_ref_dirs = {
        'violin': glob.glob('data/violin_ref/*.wav'),
        'flute' : glob.glob('data/flute_ref/*.wav'),
    }
    bg_files = glob.glob('data/background_scenes/*.wav')

    # 1) Build banks
    scales = [(128, 32), (512, 128)]
    inst_banks = build_multiscale_banks(instrument_ref_dirs, scales=scales, n_atoms_per_scale=8, device=device)
    assert 'violin' in inst_banks
    assert 'flute' in inst_banks

    # 2) Loss module
    inst_loss_module = InstrumentWaveletLoss(inst_banks, downsample_factors=[1,4], per_scale_weights=[0.6, 0.4])
    inst_loss_module.to(device)

    # 3) Dataset and Loader
    mixture_dataset = MixtureDataset(instrument_ref_dirs, bg_files, seg_len=4096, augment=False)
    loader = torch.utils.data.DataLoader(mixture_dataset, batch_size=2, shuffle=True)

    # 4) Separator
    separator = EnhancedSeparator().to(device)
    opt = torch.optim.Adam(separator.parameters(), lr=1e-4)

    # 5) Train for 1 iteration
    train_separator(separator, inst_loss_module, loader, opt, device=device, epochs=1)

    # Verify we can run prediction
    mix, target, inst_name = next(iter(loader))
    with torch.no_grad():
        pred = separator(mix.to(device))
    assert pred.shape == target.shape

def test_epistemic_agent_integration():
    from epi_sep import EpistemicAgent, SmallUNet

    device = 'cpu'
    atom_banks = {
        'violin': {'banks': [ np.random.randn(8,128).astype(np.float32), np.random.randn(8,512).astype(np.float32) ] }
    }

    separator_factory = lambda: SmallUNet(base=4)
    agent = EpistemicAgent(separator_factory, atom_banks, device=device, ensemble_size=2, tbdir='runs/test_epi')

    mix = torch.randn(1,1,4096)
    target = torch.randn(1,1,4096)

    # Test ensemble predict
    preds, mean, var, unc = agent.ensemble_predict(mix)
    assert len(preds) == 2
    assert mean.shape == mix.shape

    # Test explain and score (Mocking instrument loss for simple call)
    from epi_sep import InstrumentWaveletLoss as EpiLoss
    loss_mod = EpiLoss(atom_banks)

    score, attr = agent.explain_and_score(0, mix, mean, target, 'violin', loss_mod)
    assert 0 <= score <= 1
    assert attr.shape == mix.shape
