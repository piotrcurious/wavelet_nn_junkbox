import pytest
import torch
import numpy as np
from multiscale_wavelet_instrument_pipeline import ConvDictAE, InstrumentWaveletLoss, EnhancedSeparator, GainNorm, AdaptiveEQ, DeReverbNet, SeparatorUNet

def test_conv_dict_ae_shapes():
    n_atoms = 16
    atom_len = 128
    hop = 32
    model = ConvDictAE(n_atoms=n_atoms, atom_len=atom_len, hop=hop)

    # Input: (B, 1, T)
    x = torch.randn(2, 1, 1024)
    recon, coeff = model(x)

    assert recon.shape == x.shape
    assert coeff.shape[0] == 2
    assert coeff.shape[1] == n_atoms
    # T_out = (T_in + 2*pad - kernel) // stride + 1
    # padding is atom_len // 2 = 64
    # T_out = (1024 + 128 - 128) // 32 + 1 = 33
    assert coeff.shape[2] == 33

def test_gain_norm():
    gn = GainNorm(pool_kernel=64)
    x = torch.randn(1, 1, 1024) * 10.0
    out = gn(x)
    assert out.shape == x.shape
    # GainNorm should change the scale, not necessarily make it smaller if the input is already small,
    # but for large input it should normalized it.
    # The actual behavior depends on the untrained weights.

def test_adaptive_eq():
    eq = AdaptiveEQ(n_filters=4, kernel=17)
    x = torch.randn(1, 1, 512)
    out = eq(x)
    assert out.shape == x.shape

def test_dereverb_net():
    dr = DeReverbNet(base=8)
    x = torch.randn(1, 1, 512)
    out = dr(x)
    assert out.shape == x.shape

def test_separator_unet():
    unet = SeparatorUNet(base=8)
    x = torch.randn(1, 1, 1024)
    out = unet(x)
    assert out.shape == x.shape

def test_enhanced_separator():
    sep = EnhancedSeparator()
    # Mocking GainNorm and others for simplicity or just run the whole thing
    x = torch.randn(1, 1, 4096)
    out = sep(x)
    assert out.shape == x.shape

def test_instrument_wavelet_loss():
    # Setup mock bank
    atoms = np.random.randn(16, 128).astype(np.float32)
    inst_banks = {
        'test_inst': {
            'banks': [atoms],
            'models': [None]
        }
    }
    loss_mod = InstrumentWaveletLoss(inst_banks)

    pred = torch.randn(1, 1, 1024, requires_grad=True)
    target = torch.randn(1, 1, 1024)

    loss = loss_mod(pred, target, 'test_inst')
    assert loss.item() >= 0
    loss.backward()
    assert pred.grad is not None
