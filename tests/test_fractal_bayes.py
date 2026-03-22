import pytest
import numpy as np
import torch
from fractal_bayes import (
    dyadic_wavelet_coeffs_np,
    dyadic_wavelet_coeffs_torch,
    estimate_power_law_variance,
    compute_structure_function,
    fractal_prior_variance_from_slope,
    posterior_shrinkage_coeffs,
    FractalBayesRefiner
)

def test_dyadic_wavelet_coeffs_np():
    x = np.random.randn(1024)
    details, aJ = dyadic_wavelet_coeffs_np(x, wavelet='db4', max_level=3)
    assert len(details) == 3
    assert aJ is not None
    # total length should be approximately original
    total_len = sum(len(d) for d in details) + len(aJ)
    # Due to padding, it might be slightly larger
    assert total_len >= 1024

def test_dyadic_wavelet_coeffs_torch():
    x = torch.randn(1, 1024)
    details, aJ = dyadic_wavelet_coeffs_torch(x, wavelet='db4', max_level=3)
    assert len(details) == 3
    assert isinstance(details[0], torch.Tensor)
    assert isinstance(aJ, torch.Tensor)

def test_estimate_power_law_variance():
    # Create signals with a specific scaling
    details = [np.random.randn(100) * (2.0**j) for j in range(4)]
    slope, c, var_j, scales = estimate_power_law_variance(details)
    # var_j should be approx [1, 4, 16, 64]
    # log(var_j) = log(1) + slope * log(scale_j)
    # scales = [2, 4, 8, 16]
    # log(var_j) = [0, 1.38, 2.77, 4.15]
    # log(scales) = [0.69, 1.38, 2.07, 2.77]
    # slope should be around 2.0
    assert 1.5 < slope < 2.5

def test_compute_structure_function():
    details = [np.random.randn(100) for _ in range(3)]
    S = compute_structure_function(details, q_list=[1, 2])
    assert len(S) == 2
    assert S[0][0] == 1
    assert isinstance(S[0][1], float)

def test_fractal_prior_variance_from_slope():
    var_prior = fractal_prior_variance_from_slope(slope=1.5, J=4)
    assert len(var_prior) == 4
    assert np.all(var_prior > 0)
    # Should be increasing for positive slope
    assert var_prior[1] > var_prior[0]

def test_posterior_shrinkage_coeffs():
    pcoef = np.ones((5, 100))
    tcoef = np.ones((5, 100)) * 0.5
    prior_vars = 1.0
    post, noise_var, shrink = posterior_shrinkage_coeffs(pcoef, tcoef, prior_vars)
    assert post.shape == pcoef.shape
    assert 0 < shrink < 1

def test_fractal_bayes_refiner():
    # Mock atom banks
    atoms_scale1 = np.random.randn(8, 128)
    atoms_scale2 = np.random.randn(8, 512)
    ref_sig = np.random.randn(2048)

    atom_banks = {
        'inst1': {
            'banks': [atoms_scale1, atoms_scale2],
            'ref_signals': [ref_sig]
        }
    }

    refiner = FractalBayesRefiner(atom_banks, max_level=2)
    pred_signal = np.random.randn(2048)
    target_signal = np.random.randn(2048)

    refined, aux_loss, diagnostics = refiner.refine_prediction(
        pred_signal, target_signal, 'inst1', [atoms_scale1, atoms_scale2]
    )

    assert refined.shape == pred_signal.shape
    assert aux_loss >= 0
    assert 'slope' in diagnostics
