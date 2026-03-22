"""
fractal_bayes.py
Fractal-Bayesian hybrid inference for audio separation.

Key pieces:
 - wavelet_coeffs: compute dyadic wavelet coeffs (pywt)
 - estimate_scaling_exponent: fit power-law variance vs scale (simple fractal estimator)
 - compute_multifractal_signature: optional (structure functions tau(q))
 - fractal_prior_from_scaling: produce prior variances per scale (power-law)
 - bayesian_shrinkage_on_coeffs: posterior mean (Wiener-like shrinkage using fractal prior)
 - FractalBayesRefiner: wrapper that computes refined reconstruction + loss term
"""

import numpy as np
import pywt
import torch
import torch.nn.functional as F

# ---------- Helpers: wavelet analysis ----------
def dyadic_wavelet_coeffs_np(x, wavelet='db4', max_level=None):
    """
    x: 1D numpy signal
    returns: list of detail coeff arrays [d1, d2, ..., dJ] and approximation aJ
      where d1 is finest scale (highest freq) and dJ coarsest detail.
    Uses pywt.wavedec (dyadic).
    """
    if max_level is None:
        max_level = pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len)
    coeffs = pywt.wavedec(x, wavelet, level=max_level, mode='symmetric')
    # coeffs = [aJ, dJ, dJ-1, ..., d1] -> we will return details increasing scale order
    aJ = coeffs[0]
    details = coeffs[1:][::-1]  # reverse so details[0] = d1 (finest)
    return details, aJ

def dyadic_wavelet_coeffs_torch(x, wavelet='db4', max_level=None):
    """
    x: torch tensor (1,T) or (B,1,T) -> we handle single example here for simplicity (1,T)
    returns details as list of tensors (torch)
    """
    device = x.device
    x_np = x.detach().cpu().numpy().squeeze()
    dlist, aJ = dyadic_wavelet_coeffs_np(x_np, wavelet=wavelet, max_level=max_level)
    return [torch.from_numpy(d.astype(np.float32)).to(device) for d in dlist], torch.from_numpy(aJ.astype(np.float32)).to(device)

# ---------- Fractal / scaling estimation ----------
def estimate_power_law_variance(details_list):
    """
    Given list of detail coeff arrays [d1, d2, ..., dJ] (each 1D numpy),
    compute var_j = variance(|d_j|) and fit log(var_j) = c + s*log(scale_j)
    dyadic scales: scale_j = 2^j (j=1..J)
    returns slope s and intercept c (where slope relates to fractal H)
    """
    var_j = np.array([np.var(np.abs(d)) + 1e-12 for d in details_list])
    J = len(var_j)
    scales = np.array([2.0**(j+1) for j in range(J)])  # j=0->scale 2^1 for d1
    logv = np.log(var_j)
    logs = np.log(scales)
    A = np.vstack([logs, np.ones_like(logs)]).T
    s, c = np.linalg.lstsq(A, logv, rcond=None)[0]
    return float(s), float(c), var_j, scales

# ---------- Multifractal signature (optional, structure function tau(q)) ----------
def compute_structure_function(details_list, q_list=[-2,-1,0,1,2]):
    """
    Compute S(q,j) = mean(|d_j|^q) for each scale j, then fit tau(q)
    returns tau(q) approx vector where tau(q) = slope log2 S(q,j) vs j
    """
    J = len(details_list)
    S = []
    for q in q_list:
        vals = np.array([np.mean(np.abs(d)**q) for d in details_list])
        # regress log2(vals) vs j
        js = np.arange(1, J+1)
        logvals = np.log2(vals + 1e-12)
        A = np.vstack([js, np.ones_like(js)]).T
        slope, intercept = np.linalg.lstsq(A, logvals, rcond=None)[0]
        S.append((q, float(slope)))  # slope is tau(q)
    # return list of (q, tau(q))
    return S

# ---------- Fractal-driven prior ----------
def fractal_prior_variance_from_slope(slope, base_scale_variance=1.0, J=6):
    """
    Given slope from log(var) vs log(scale), create prior variance per dyadic scale j:
    model var_prior(scale_j) = base_scale_variance * scale_j^{slope}
    return numpy array var_prior[j] for j=1..J
    """
    scales = np.array([2.0**(j+1) for j in range(J)])  # j indexing consistent
    var_prior = base_scale_variance * (scales ** slope)
    # normalize to unit mean to avoid huge magnitudes
    var_prior = var_prior / (np.mean(var_prior) + 1e-12)
    return var_prior

# ---------- Bayesian shrinkage on coefficients ----------
def posterior_shrinkage_coeffs(pcoef, tcoef, prior_variances, noise_variance_est=None):
    """
    pcoef, tcoef: numpy arrays shaped (n_atoms, Tcoef) or (n_channels, n_atoms, T)
      We operate per-scale. Simpler: assume pcoef is shape (n_atoms, Tpos).
    prior_variances: scalar or vector (per-atom or per-scale) giving prior var of coefficient values
    noise_variance_est: estimated observation noise variance (scalar)
    We implement a Wiener-like shrinkage posterior mean:
      post_mean = (prior_var / (prior_var + noise_var)) * observed_coef
    Returns posterior coefficients array of same shape.
    """
    if noise_variance_est is None:
        # estimate noise variance from (pcoef - tcoef) if target available else from pcoef median-abs-deviation
        if tcoef is not None:
            resid = pcoef - tcoef
            noise_variance_est = np.mean(resid**2)
        else:
            # robust MAD
            noise_variance_est = (np.median(np.abs(pcoef - np.median(pcoef))) / 0.6745)**2 + 1e-12
    # ensure shapes broadcast: prior_variances can be scalar or (n_atoms,) or (1,n_atoms,1)
    prior = np.asarray(prior_variances)
    # broadcast prior to pcoef shape
    while prior.ndim < pcoef.ndim:
        prior = prior[..., None]
    shrinkage = prior / (prior + noise_variance_est + 1e-12)
    post = shrinkage * pcoef
    return post, noise_variance_est, shrinkage

# ---------- Combine into a refiner class ----------
class FractalBayesRefiner:
    """
    High-level wrapper: given instrument atoms (per-scale) and a predicted signal,
    compute wavelet/atom coefficients, estimate fractal slope from reference signals,
    do Bayesian shrinkage on predicted coefficients (fractal prior), reconstruct refined signal.
    Also computes an L2 loss term between refined and target to use as auxiliary loss.
    """
    def __init__(self, atom_banks, wavelet='db4', max_level=6, device='cpu'):
        """
        atom_banks: dict {inst_name: {'banks':[np.atoms_scale1, atoms_scale2, ...], 'ref_signals':[list of numpy signals for fractal estimation]}}
        atoms_scale: numpy array (n_atoms, atom_len)
        """
        self.atom_banks = atom_banks
        self.wavelet = wavelet
        self.max_level = max_level
        self.device = device

    def estimate_fractal_prior_for_inst(self, inst_name):
        """
        Estimate slope from provided ref_signals for instrument inst_name.
        Returns prior_variances_per_scale (numpy vector len=J)
        """
        ref_sig_list = self.atom_banks[inst_name].get('ref_signals', [])
        # compute average slope across reference list
        slopes = []
        for s in ref_sig_list:
            dlist, aJ = dyadic_wavelet_coeffs_np(s, wavelet=self.wavelet, max_level=self.max_level)
            s_slope, c, var_j, scales = estimate_power_law_variance(dlist)
            slopes.append(s_slope)
        if len(slopes) == 0:
            slope = -1.0  # fallback conservative value
        else:
            slope = float(np.median(slopes))
        # produce prior variance per scale
        var_prior = fractal_prior_variance_from_slope(slope, base_scale_variance=1.0, J=self.max_level)
        return var_prior, slope

    def refine_prediction(self, pred_signal, target_signal, inst_name, atoms_scale_list):
        """
        pred_signal, target_signal: 1D numpy arrays (same length)
        atoms_scale_list: list of atom matrices corresponding to scales (n_atoms, atom_len)
        returns refined_signal (1D numpy), auxiliary_loss (float), diagnostics dict
        """
        # 1) compute dyadic wavelet details to get number of scales J
        dlist_pred, _ = dyadic_wavelet_coeffs_np(pred_signal, wavelet=self.wavelet, max_level=self.max_level)
        J = len(dlist_pred)
        # 2) compute fractal prior var per scale from ref_signals
        var_prior_scales, slope = self.estimate_fractal_prior_for_inst(inst_name)
        var_prior_scales = var_prior_scales[:J]
        # 3) for each scale, compute atom convolution coefficients (here we use conv with atoms as analysis)
        # We'll do a simplified approach: downsample pred/target to approximate scale and convolve with atoms
        posterior_coeffs_scales = []
        residual_noise_vars = []
        shrinkage_gains = []
        recon_parts = []
        for si, atoms in enumerate(atoms_scale_list[:J]):
            # atoms: (n_atoms, atom_len)
            kernel = atoms[:, None, :]  # shape (n_atoms,1,L)
            # convert to convolution using numpy via FFT conv for speed, but here keep simple using np.convolve per atom & position
            # We'll compute coefficients by convolving pred with each atom (full conv) and cropping to original length
            n_atoms, L = atoms.shape
            pcoef = np.stack([np.convolve(pred_signal, atoms[k, ::-1], mode='same') for k in range(n_atoms)], axis=0)
            tcoef = np.stack([np.convolve(target_signal, atoms[k, ::-1], mode='same') for k in range(n_atoms)], axis=0)
            # compute shrinkage for this scale using prior variance = var_prior_scales[si]
            prior_var = var_prior_scales[si]
            # use per-scale prior; optionally scale per-atom by their RMS in reference data (not implemented here for brevity)
            post, noise_var, shrink = posterior_shrinkage_coeffs(pcoef, tcoef, prior_variances=prior_var)
            # reconstruct partial signal from posterior coefficients using transpose convolution (atoms as decoders)
            # reconstruction = sum_k conv(post_k, atom_k) -> approximate by full conv
            recon_scale = np.sum([np.convolve(post[k, :], atoms[k, :], mode='same') for k in range(n_atoms)], axis=0)
            posterior_coeffs_scales.append(post)
            residual_noise_vars.append(noise_var)
            shrinkage_gains.append(shrink)
            recon_parts.append(recon_scale)
        # combine recon_parts (simple sum)
        refined = np.sum(recon_parts, axis=0)
        # optionally add residual approximation (coarsest aJ) but omitted here
        # compute auxiliary loss L2 between refined and target
        aux_loss = float(np.mean((refined - target_signal)**2))
        diagnostics = {
            'slope': slope,
            'var_prior_scales': var_prior_scales.tolist(),
            'residual_noise_vars': [float(v) for v in residual_noise_vars],
            'mean_shrinkage': float(np.mean([np.mean(s) for s in shrinkage_gains]))
        }
        return refined, aux_loss, diagnostics
