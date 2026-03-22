"""
fractal_bayes.py
Fractal-Bayesian hybrid inference for audio separation.

This module provides tools for:
1. Feature Extraction: Multi-scale wavelet analysis.
2. Prior Estimation: Estimating fractal scaling (power-law) from reference signals.
3. Post-processing Refinement: Bayesian shrinkage of wavelet coefficients.

Note: This implementation primarily uses NumPy/PyWavelets and is intended for
post-processing or non-differentiable inference. Gradients will NOT flow through
the FractalBayesRefiner or any analysis functions in this module.
"""

import numpy as np
import pywt
import torch

# ---------- 1. Feature Extraction ----------

def analyze_wavelet_coeffs(x, wavelet='db4', max_level=None):
    """
    Decompose signal into wavelet coefficients.
    x: 1D numpy array.
    returns: details [d1, d2, ..., dJ] and approximation aJ.
    """
    if x.ndim > 1:
        x = x.ravel()

    # Validate signal length
    min_len = pywt.Wavelet(wavelet).dec_len
    if len(x) < min_len:
        return [], x

    max_l = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    if max_level is None:
        level = max_l
    else:
        level = min(max_level, max_l)

    if level <= 0:
        return [], x

    coeffs = pywt.wavedec(x, wavelet, level=level, mode='symmetric')
    # coeffs = [aJ, dJ, dJ-1, ..., d1]
    aJ = coeffs[0]
    details = coeffs[1:][::-1] # [d1, d2, ..., dJ] where d1 is finest
    return details, aJ

def analyze_wavelet_coeffs_torch(x, wavelet='db4', max_level=None):
    """
    Torch-wrapped version of analyze_wavelet_coeffs.
    Note: Still uses NumPy internally; gradients do not flow.
    """
    device = x.device
    x_np = x.detach().cpu().numpy()
    details_np, aJ_np = analyze_wavelet_coeffs(x_np, wavelet, max_level)

    details = [torch.from_numpy(d.astype(np.float32)).to(device) for d in details_np]
    aJ = torch.from_numpy(aJ_np.astype(np.float32)).to(device)
    return details, aJ

# ---------- 2. Prior Estimation ----------

def estimate_fractal_scaling(details_list):
    """
    Estimate power-law variance slope: log(var) = c + s * log(scale).
    details_list: [d1, d2, ..., dJ]
    returns: slope (s), intercept (c).
    """
    if not details_list:
        return 0.0, 0.0

    # Compute variance per scale
    # Weighted regression: coarser scales have fewer coefficients, so we give them less weight
    variances = []
    weights = []
    scales = []

    for j, d in enumerate(details_list):
        v = np.var(d) + 1e-12
        variances.append(v)
        scales.append(2.0**(j+1))
        weights.append(np.sqrt(len(d))) # weight proportional to sqrt of sample size

    log_v = np.log(variances)
    log_s = np.log(scales)

    # Weighted least squares
    W = np.diag(weights)
    A = np.vstack([log_s, np.ones_like(log_s)]).T
    # Solve (A^T W A) x = A^T W b
    try:
        sol = np.linalg.lstsq(W @ A, W @ log_v, rcond=None)[0]
        s, c = sol
    except:
        s, c = 0.0, 0.0

    return float(s), float(c)

def compute_multifractal_signature(details_list, q_list=[-2, -1, 0, 1, 2], eps=1e-8):
    """
    Compute structure function tau(q) describing scaling of q-th moments.
    """
    if not details_list:
        return []

    J = len(details_list)
    out = []
    for q in q_list:
        # Use log-domain for stability with negative q
        # S(q, j) = mean(|d_j|^q)
        # Avoid exploding values by clipping abs(d) from below
        moments = []
        for d in details_list:
            m = np.mean((np.abs(d) + eps) ** q)
            moments.append(m)

        js = np.arange(1, J + 1)
        log_m = np.log2(np.array(moments) + eps)
        A = np.vstack([js, np.ones_like(js)]).T
        slope, _ = np.linalg.lstsq(A, log_m, rcond=None)[0]
        out.append((q, float(slope)))
    return out

# ---------- 3. Post-processing Refinement ----------

def apply_bayesian_shrinkage(coeffs, prior_var, noise_var):
    """
    Perform Wiener-like shrinkage: E[s|x] = (v_s / (v_s + v_n)) * x
    """
    shrinkage = prior_var / (prior_var + noise_var + 1e-12)
    return shrinkage * coeffs, shrinkage

class FractalBayesRefiner:
    def __init__(self, wavelet='db4', max_level=6):
        self.wavelet = wavelet
        self.max_level = max_level
        self.reference_priors = {} # Store slopes per instrument

    def train_prior(self, inst_name, ref_signals):
        """
        Learn typical fractal slope for an instrument.
        ref_signals: list of 1D numpy arrays.
        """
        slopes = []
        for s in ref_signals:
            details, _ = analyze_wavelet_coeffs(s, self.wavelet, self.max_level)
            slope, _ = estimate_fractal_scaling(details)
            slopes.append(slope)

        if slopes:
            self.reference_priors[inst_name] = np.median(slopes)
        else:
            self.reference_priors[inst_name] = -1.0 # fallback

    def refine(self, pred_signal, inst_name, target_signal=None):
        """
        Apply fractal-driven Bayesian shrinkage to the prediction.
        target_signal: if provided, only used for auxiliary loss computation (not for shrinkage).
        """
        # 1. Analyze
        details, aJ = analyze_wavelet_coeffs(pred_signal, self.wavelet, self.max_level)
        if not details:
            return pred_signal, 0.0, {'status': 'too_short'}

        J = len(details)

        # 2. Get prior scaling
        slope = self.reference_priors.get(inst_name, -1.0)
        # Prior variance model: var(scale) = base * scale^slope
        # We can estimate base from the overall energy of pred_signal
        scales = np.array([2.0**(j+1) for j in range(J)])
        prior_vars = scales ** slope
        # Normalize prior_vars to have similar scale as details
        total_detail_var = np.mean([np.var(d) for d in details])
        prior_vars = prior_vars * (total_detail_var / (np.mean(prior_vars) + 1e-12))

        # 3. Estimate noise variance
        # Standard wavelet heuristic: estimate noise from d1 (finest scale)
        # using Robust Median Absolute Deviation (MAD)
        d1 = details[0]
        noise_std = np.median(np.abs(d1)) / 0.6745
        noise_var = noise_std**2

        # 4. Shrink
        refined_details = []
        shrinkage_stats = []
        for j in range(J):
            refined_d, shrink = apply_bayesian_shrinkage(details[j], prior_vars[j], noise_var)
            refined_details.append(refined_d)
            shrinkage_stats.append(float(np.mean(shrink)))

        # 5. Synthesize
        # waverec expects [aJ, dJ, dJ-1, ..., d1]
        rec_coeffs = [aJ] + refined_details[::-1]
        refined = pywt.waverec(rec_coeffs, self.wavelet, mode='symmetric')

        # Match length
        if len(refined) > len(pred_signal):
            refined = refined[:len(pred_signal)]
        elif len(refined) < len(pred_signal):
            refined = np.pad(refined, (0, len(pred_signal) - len(refined)))

        # 6. Metrics
        aux_loss = 0.0
        if target_signal is not None:
            if target_signal.shape == refined.shape:
                aux_loss = float(np.mean((refined - target_signal)**2))

        diagnostics = {
            'slope': slope,
            'noise_var': float(noise_var),
            'avg_shrinkage': float(np.mean(shrinkage_stats)),
            'nans_detected': bool(np.isnan(refined).any() or np.isinf(refined).any())
        }

        return refined, aux_loss, diagnostics
