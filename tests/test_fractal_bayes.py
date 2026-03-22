import pytest
import numpy as np
import torch
from fractal_bayes import (
    analyze_wavelet_coeffs,
    analyze_wavelet_coeffs_torch,
    estimate_fractal_scaling,
    compute_multifractal_signature,
    FractalBayesRefiner
)

def test_analyze_wavelet_coeffs():
    x = np.random.randn(1024)
    details, aJ = analyze_wavelet_coeffs(x, wavelet='db4', max_level=3)
    assert len(details) == 3
    assert aJ is not None

def test_analyze_wavelet_coeffs_too_short():
    x = np.random.randn(5) # Very short
    details, aJ = analyze_wavelet_coeffs(x, wavelet='db4')
    assert len(details) == 0
    assert len(aJ) == 5

def test_analyze_wavelet_coeffs_torch():
    x = torch.randn(1, 1024)
    details, aJ = analyze_wavelet_coeffs_torch(x, wavelet='db4', max_level=3)
    assert len(details) == 3
    assert isinstance(details[0], torch.Tensor)

def test_estimate_fractal_scaling():
    # Synthetic scaling
    details = [np.random.randn(1000 // (2**j)) * (2.0**(-0.5*j)) for j in range(4)]
    slope, c = estimate_fractal_scaling(details)
    assert isinstance(slope, float)

def test_compute_multifractal_signature_stability():
    # Negative q with tiny values
    details = [np.random.randn(100) * 1e-15 for _ in range(3)]
    sig = compute_multifractal_signature(details, q_list=[-2, 2])
    assert len(sig) == 2
    assert not np.isnan(sig[0][1])
    assert not np.isinf(sig[0][1])

def test_fractal_bayes_refiner_flow():
    refiner = FractalBayesRefiner(max_level=3)
    ref_sigs = [np.random.randn(2048) for _ in range(3)]
    refiner.train_prior('violin', ref_sigs)

    pred = np.random.randn(2048)
    target = np.random.randn(2048)

    refined, aux_loss, diag = refiner.refine(pred, 'violin', target_signal=target)

    assert refined.shape == pred.shape
    assert aux_loss >= 0
    assert 'noise_var' in diag
    assert not diag['nans_detected']

def test_refiner_mismatched_target():
    refiner = FractalBayesRefiner()
    pred = np.random.randn(1024)
    target = np.random.randn(512) # mismatched

    refined, aux_loss, diag = refiner.refine(pred, 'inst', target_signal=target)
    assert aux_loss == 0.0 # Should skip calculation safely
