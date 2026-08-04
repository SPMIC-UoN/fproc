"""
magorino_fit.py
================
Python implementation of MAGO (Gaussian magnitude) and MAGORINO (Rician magnitude)
multi-echo fat-water fitting for R2* and PDFF quantification.

Based on:
  - Bray et al. MRM 2023 (MAGORINO): https://doi.org/10.1002/mrm.29493
  - Hernando et al. MRM 2013:         https://doi.org/10.1002/mrm.27728

Signal model
------------
Multi-peak fat spectrum (Hernando et al. multisite) with a single R2* for fat + water:

  S(t) = [W + F * sum_k(a_k * exp(i*2pi*f_k*t))] * exp(-R2* * t) * exp(i*2pi*fB*t)

where f_k and a_k are the fat peak frequencies (Hz) and relative amplitudes.

Fitting pipeline
----------------
1.  Estimate sigma (noise SD) from an ROI using step-1 Rician fit with floating sigma.
2.  Smooth / use that sigma estimate.
3.  Run per-voxel MAGO (Gaussian) and MAGORINO (Rician) fitting with dual
    initialisation (water-dominant / fat-dominant) and pick the best solution.
4.  Post-process with spatially filtered likelihood-difference map to resolve
    fat-water ambiguity (MAGORINO step).
5.  Save output maps as NIfTI and display results.

Usage
-----
  python magorino_fit.py [--input path/to/t2star.nii.gz] [--slice N] [--method all|gaussian|rician]

The script also works when imported as a module — call fit_image() directly.
"""

import argparse
import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.special import i0   # modified Bessel function I_0

from fproc.module import Module

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GPU detection — use CuPy if available, fall back to NumPy silently
# ---------------------------------------------------------------------------
def _detect_gpu():
    """Return (xp, gpu_available) where xp is cupy or numpy."""
    try:
        import cupy as cp  # type: ignore[import-not-found]
        cp.array([1.0])          # force a real CUDA call to confirm it works
        return cp, True
    except Exception:
        return np, False

_xp, _GPU_AVAILABLE = _detect_gpu()


# ---------------------------------------------------------------------------
# 1.  Multi-peak fat signal model (Hernando multisite spectrum, 3T / nT)
# ---------------------------------------------------------------------------

# Relative amplitudes and chemical shifts (ppm) from Hernando et al.
_FAT_AMPS   = np.array([0.087, 0.694, 0.128, 0.004, 0.039, 0.048])
_FAT_SHIFTS = np.array([-3.9, -3.5, -2.7, -2.04, -0.49, 0.50])   # ppm
_GYRO       = 42.58e6   # Hz/T (proton gyromagnetic ratio)


def multi_peak_fat_signal(t_ms: np.ndarray,
                           tesla: float,
                           F: float,
                           W: float,
                           R2star: float,
                           fB: float = 0.0) -> np.ndarray:
    """
    Complex multi-peak fat-water signal at echo times t_ms.

    Parameters
    ----------
    t_ms   : (nTE,) echo times in milliseconds
    tesla  : field strength in Tesla
    F      : fat amplitude (arbitrary units)
    W      : water amplitude (arbitrary units)
    R2star : R2* in ms^-1  (= 1/T2* in ms)
    fB     : B0 field inhomogeneity offset in Hz (default 0)

    Returns
    -------
    S : (nTE,) complex signal array
    """
    larmor = tesla * _GYRO                              # Hz
    fat_freqs_hz   = _FAT_SHIFTS * larmor * 1e-6       # Hz  (ppm × Hz/ppm)
    fat_freqs_cyms = fat_freqs_hz / 1000.0             # cycles / ms
    fat_angular    = 2 * np.pi * fat_freqs_cyms        # rad / ms

    water_angular  = 0.0                               # water reference at 0 ppm

    # Sum fat peaks
    S = np.zeros(len(t_ms), dtype=complex)
    for k in range(len(_FAT_AMPS)):
        S += F * _FAT_AMPS[k] * np.exp(1j * fat_angular[k] * t_ms)
    S += W * np.exp(1j * water_angular * t_ms)

    # R2* decay
    S *= np.exp(-R2star * t_ms)

    # B0 inhomogeneity
    fB_cyms = fB / 1000.0
    S *= np.exp(1j * 2 * np.pi * fB_cyms * t_ms)

    return S


# ---------------------------------------------------------------------------
# 2.  Log-likelihood functions
# ---------------------------------------------------------------------------

def _log_besseli0(x: np.ndarray) -> np.ndarray:
    """
    Fast vectorised log(I_0(x)).

    For large arrays (>10k elements) uses the asymptotic expansion throughout:
        log I_0(x) ≈ x - 0.5 * log(2*pi*x)

    For small arrays uses scipy for accuracy in the x < 3.75 region.
    CuPy path always uses the asymptotic form.
    """
    xp = _xp if hasattr(x, 'device') else np
    ax = xp.abs(x).astype(np.float32 if x.dtype == np.float32 else np.float64)
    safe_ax = xp.where(ax > 1e-6, ax, xp.full_like(ax, 1e-6))
    log_asymp = ax - 0.5 * xp.log(2.0 * np.pi * safe_ax)

    if xp is not np or ax.size > 10_000:
        return log_asymp

    small = ax <= 3.75
    if small.any():
        result = log_asymp.copy()
        result[small] = np.log(np.clip(i0(ax[small]), 1e-30, None))
        return result
    return log_asymp


def gaussian_log_lik(measured: np.ndarray,
                     predicted: np.ndarray,
                     sigma: float) -> float:
    """Gaussian (MAGO) log-likelihood for magnitude data."""
    sq_errors = (measured - predicted) ** 2
    return float(-0.5 * np.sum(np.log(2 * np.pi * sigma ** 2) + sq_errors / sigma ** 2))


def rician_log_lik(measured: np.ndarray,
                   predicted: np.ndarray,
                   sigma: float) -> float:
    """Rician (MAGORINO) log-likelihood for magnitude data."""
    sigma2 = sigma ** 2
    sumsqsc = (measured ** 2 + predicted ** 2) / (2 * sigma2)
    scp = measured * predicted / sigma2
    lb0 = _log_besseli0(scp)
    logliks = np.log(measured + 1e-30) - np.log(sigma2) - sumsqsc + lb0
    return float(np.sum(logliks))


# ---------------------------------------------------------------------------
# 3.  Per-voxel fitting
# ---------------------------------------------------------------------------

def precompute_signal_shapes(t_ms: np.ndarray,
                             tesla: float,
                             n_pdff: int = 101,
                             n_r2: int = 1001,
                             r2_max: float = 0.4):
    """
    Vectorised precomputation of all (PDFF, R2*) signal shapes in one pass.

    Default grid: 101 PDFF steps × 1001 R2* steps over 0–0.3 ms⁻¹
    (= 0–300 s⁻¹), giving 0.3 s⁻¹ resolution — sufficient for liver
    (typical range 20–120 s⁻¹) and iron-overloaded liver (up to ~300 s⁻¹).
    Use r2_max=1.0 for very high iron / skeletal muscle if needed.

    Returns
    -------
    shapes    : (n_pdff, n_r2, nTE) magnitude signal with S0=1
    pdff_grid : (n_pdff,) PDFF values 0..1
    r2_grid   : (n_r2,)   R2* values in ms^-1
    """
    larmor       = tesla * _GYRO
    fat_freqs_hz = _FAT_SHIFTS * larmor * 1e-6          # Hz
    fat_angular  = 2 * np.pi * fat_freqs_hz / 1000.0    # rad/ms

    # Fat phasor sum at each TE:  sum_k a_k * exp(i*w_k*t)  →  (nTE,)
    fat_phasors  = np.sum(
        _FAT_AMPS[:, np.newaxis] *
        np.exp(1j * fat_angular[:, np.newaxis] * t_ms[np.newaxis, :]),
        axis=0)

    pdff_grid    = np.linspace(0.0, 1.0, n_pdff)
    r2_grid      = np.linspace(0.0, r2_max, n_r2)

    # complex signal (S0=1): [pdff*(fat-1)+1] * exp(-r2*t)
    # shape: (n_pdff, nTE)  ×  (n_r2, nTE)  →  broadcast to (n_pdff, n_r2, nTE)
    complex_base = (pdff_grid[:, np.newaxis] * (fat_phasors[np.newaxis, :] - 1.0)
                    + 1.0)                               # (n_pdff, nTE)
    decay        = np.exp(-r2_grid[:, np.newaxis] * t_ms[np.newaxis, :])  # (n_r2, nTE)
    signal       = complex_base[:, np.newaxis, :] * decay[np.newaxis, :, :]  # (n_pdff,n_r2,nTE)

    return np.abs(signal), pdff_grid, r2_grid


def _fit_voxel_core(S_mag: np.ndarray,
                    shapes: np.ndarray,
                    pdff_grid: np.ndarray,
                    r2_grid: np.ndarray,
                    sigma: float,
                    method: str = 'gaussian') -> dict:
    """
    Core per-voxel fit given pre-computed shapes (n_pdff, n_r2, nTE).

    Gaussian:  S0 is solved in closed form (least-squares scale factor),
               then the full log-likelihood grid is computed in one einsum.

    Rician:    Same S0_ls from Gaussian step, then a single 1-D bounded
               minimisation over S0 only at the best (PDFF, R2*) grid point
               (shapes are fixed at that point, so S0 is the only unknown).
    """
    # --- Closed-form S0 and Gaussian LL grid (always computed) ---
    num    = np.einsum('ijk,k->ij', shapes, S_mag)          # (n_pdff, n_r2)
    denom  = np.einsum('ijk,ijk->ij', shapes, shapes)        # (n_pdff, n_r2)
    S0_ls  = np.clip(num / (denom + 1e-30), 0.0, None)

    S_pred = S0_ls[:, :, np.newaxis] * shapes                # (n_pdff, n_r2, nTE)
    resid2 = (S_mag - S_pred) ** 2
    gauss_ll = -0.5 * np.sum(
        np.log(2 * np.pi * sigma ** 2) + resid2 / sigma ** 2, axis=2)  # (n_pdff, n_r2)

    if method == 'gaussian':
        ll_grid = gauss_ll
    else:
        # --- Rician LL grid at Gaussian S0_ls ---
        sigma2  = sigma ** 2
        scp     = np.clip(S_mag * S_pred / sigma2, 0, None)
        sumsq   = (S_mag ** 2 + S_pred ** 2) / (2 * sigma2)
        lb0     = _log_besseli0(scp)
        ll_grid = np.sum(
            np.log(np.maximum(S_mag, 1e-30)) - np.log(sigma2) - sumsq + lb0,
            axis=2)                                           # (n_pdff, n_r2)

    # --- Global best ---
    bi, ri   = np.unravel_index(np.argmax(ll_grid), ll_grid.shape)
    best_pdff = float(pdff_grid[bi])
    best_r2   = float(r2_grid[ri])
    best_S0   = float(S0_ls[bi, ri])
    best_ll   = float(ll_grid[bi, ri])

    # --- Refine S0 for Rician at the best (PDFF, R2*) ---
    if method == 'rician':
        from scipy.optimize import minimize_scalar
        shape_b = shapes[bi, ri]
        lo = max(best_S0 * 0.5, 0.0)
        hi = best_S0 * 1.5 + float(np.max(S_mag)) * 0.05 + 1.0
        res = minimize_scalar(
            lambda s: -rician_log_lik(S_mag, max(s, 0.0) * shape_b, sigma),
            bounds=(lo, hi), method='bounded', options={'xatol': 0.5, 'maxiter': 30})
        best_S0 = max(float(res.x), 0.0)
        best_ll = float(-res.fun)

    # --- Second-best (PDFF region ≥0.15 away, for likelihood-diff map) ---
    far_mask = np.abs(pdff_grid[:, np.newaxis] - best_pdff) > 0.15
    if far_mask.any():
        ll_far  = np.where(far_mask, ll_grid, -np.inf)
        bi2, ri2 = np.unravel_index(np.argmax(ll_far), ll_far.shape)
        sec_pdff = float(pdff_grid[bi2])
        sec_r2   = float(r2_grid[ri2])
        sec_S0   = float(S0_ls[bi2, ri2])
        sec_ll   = float(ll_far[bi2, ri2])
    else:
        sec_pdff, sec_r2, sec_S0, sec_ll = best_pdff, best_r2, best_S0, best_ll

    F_b = best_pdff * best_S0;  W_b = (1 - best_pdff) * best_S0
    F_s = sec_pdff  * sec_S0;   W_s = (1 - sec_pdff)  * sec_S0

    return dict(F=F_b, W=W_b, R2=best_r2, PDFF=best_pdff,
                S0=best_S0, loglik=best_ll,
                loglik_opt1=best_ll, loglik_opt2=sec_ll,
                F_opt1=F_b, W_opt1=W_b, R2_opt1=best_r2,
                F_opt2=F_s, W_opt2=W_s, R2_opt2=sec_r2)


def _neg_gaussian_obj_simple(S_mag, S_pred, sigma):
    return -gaussian_log_lik(S_mag, S_pred, sigma)


def _neg_rician_obj_simple(S_mag, S_pred, sigma):
    return -rician_log_lik(S_mag, S_pred, sigma)


def fit_voxel_gaussian(t_ms: np.ndarray,
                       tesla: float,
                       S_mag: np.ndarray,
                       sigma: float) -> dict:
    """MAGO: Gaussian magnitude fit for a single voxel (precomputes shapes internally)."""
    shapes, pdff_grid, r2_grid = precompute_signal_shapes(t_ms, tesla)
    return _fit_voxel_core(S_mag, shapes, pdff_grid, r2_grid, sigma, method='gaussian')


def fit_voxel_rician(t_ms: np.ndarray,
                     tesla: float,
                     S_mag: np.ndarray,
                     sigma: float) -> dict:
    """MAGORINO: Rician magnitude fit for a single voxel (precomputes shapes internally)."""
    shapes, pdff_grid, r2_grid = precompute_signal_shapes(t_ms, tesla)
    return _fit_voxel_core(S_mag, shapes, pdff_grid, r2_grid, sigma, method='rician')


# ---------------------------------------------------------------------------
# 4.  Sigma estimation from ROI (step 1 of MAGORINO)
# ---------------------------------------------------------------------------

def estimate_sigma_from_roi(t_ms: np.ndarray,
                             tesla: float,
                             S_mag_roi: np.ndarray) -> float:
    """
    Estimate noise sigma from an ROI by fitting a representative voxel/ROI
    signal with floating sigma, scanning PDFF on a coarse grid (since PDFF
    and sigma can otherwise trade off against each other) and refining R2*,
    S0 and sigma at each grid point.

    S_mag_roi : (nTE,) median or mean signal from ROI voxels
    """
    rough_sigma = max(np.percentile(S_mag_roi, 10) * 0.1, 1e-3)
    S0_init = np.max(S_mag_roi)

    best_ll = -np.inf
    best_sigma = rough_sigma

    for pdff in np.linspace(0.0, 1.0, 21):
        F_shape, W_shape = pdff, 1 - pdff

        def neg_ll(params):
            R2, S0, sig = params
            if R2 < 0 or S0 < 0 or sig <= 0:
                return 1e10
            shape = np.abs(multi_peak_fat_signal(t_ms, tesla, F_shape, W_shape, R2, 0.0))
            S_pred = S0 * shape
            return -rician_log_lik(S_mag_roi, S_pred, sig)

        res = minimize(neg_ll, [0.05, S0_init, rough_sigma],
                       bounds=[(0, 2), (0, 5*S0_init), (rough_sigma*0.05, rough_sigma*10)],
                       method='L-BFGS-B', options={'maxiter': 150, 'ftol': 1e-9})
        ll = -res.fun
        if ll > best_ll:
            best_ll = ll
            best_sigma = abs(res.x[2])

    return best_sigma


# ---------------------------------------------------------------------------
# 5.  Image-level fitting (slice or volume)
# ---------------------------------------------------------------------------

def _fit_slice_vectorised(slice_mag: np.ndarray,
                           shapes: np.ndarray,
                           pdff_grid: np.ndarray,
                           r2_grid: np.ndarray,
                           sigma: float,
                           method: str,
                           mask: np.ndarray,
                           xp) -> dict:
    """
    Fit all masked voxels in a slice simultaneously using batched array ops.

    Batch size is chosen automatically to use ~25% of available RAM/VRAM,
    so this works on any machine. On GPU the batch is limited by VRAM.
    """
    nX, nY, nTE  = slice_mag.shape
    n_pdff, n_r2 = shapes.shape[:2]
    nPR          = n_pdff * n_r2

    LOG.info(f"Fitting slice {slice_mag.shape} with {nPR} (PDFF,R2*) combinations, method={method}, sigma={sigma:.3f}")
    # Auto batch size: Gaussian needs ~3 arrays of shape (B, nPR) in float32
    try:
        import psutil
        avail_bytes = psutil.virtual_memory().available
    except ImportError:
        avail_bytes = 2 * 1024**3   # fallback: assume 2 GB free
    bytes_per_vox = 3 * nPR * 4    # 3 float32 arrays of length nPR per voxel
    batch_size = max(1, int(avail_bytes * 0.25 / bytes_per_vox))
    batch_size = min(batch_size, 1024)   # cap to avoid excessive single-batch time
    LOG.info(f"Auto batch size: {batch_size} voxels per batch (available RAM: {avail_bytes/1e9:.2f} GB)")

    sl_all  = slice_mag.reshape(-1, nTE).astype(np.float32)   # (nV, nTE)
    msk_all = mask.reshape(-1).astype(bool)                    # (nV,)
    nV      = sl_all.shape[0]

    # Pre-transfer shapes to GPU once
    sh     = xp.asarray(shapes.astype(np.float32))            # (nP, nR, nTE)
    sh_flat = sh.reshape(-1, nTE)                              # (nP*nR, nTE)
    denom  = xp.einsum('ij,ij->i', sh_flat, sh_flat)          # (nP*nR,)
    pdff_arr = xp.asarray(pdff_grid.astype(np.float32))       # (nP,)
    r2_arr   = xp.asarray(r2_grid.astype(np.float32))         # (nR,)
    sigma2   = float(sigma) ** 2

    # Output arrays (CPU)
    best_pdff_all = np.zeros(nV, dtype=np.float32)
    best_r2_all   = np.zeros(nV, dtype=np.float32)
    best_S0_all   = np.zeros(nV, dtype=np.float32)
    best_ll_all   = np.full(nV, -np.inf, dtype=np.float32)
    sec_pdff_all  = np.zeros(nV, dtype=np.float32)
    sec_r2_all    = np.zeros(nV, dtype=np.float32)
    sec_ll_all    = np.full(nV, -np.inf, dtype=np.float32)

    for start in range(0, nV, batch_size):
        end     = min(start + batch_size, nV)
        sl_b    = xp.asarray(sl_all[start:end])               # (B, nTE)
        msk_b   = msk_all[start:end]                           # (B,) bool, CPU
        B       = sl_b.shape[0]
        LOG.info(f"Processing batch {start}:{end} ({B} voxels)")

        # S0: (B, nP*nR)
        num  = sl_b @ sh_flat.T
        S0   = xp.clip(num / (denom[xp.newaxis, :] + 1e-30), 0, None)
        # Zero out background
        msk_gpu = xp.asarray(msk_b.astype(np.float32))
        S0 = S0 * msk_gpu[:, xp.newaxis]

        # Gaussian LL without materialising (B, nP*nR, nTE):
        # RSS = ||sl - S0*sh||^2 = ||sl||^2 - S0^2 * ||sh||^2   (by LS property)
        # => LL = -nTE/2*log(2pi*sigma2) - RSS/(2*sigma2)
        if method == 'gaussian':
            sl_ss   = xp.einsum('bi,bi->b', sl_b, sl_b)              # (B,) ||sl||^2
            sh_ss   = denom                                            # (nP*nR,) ||sh||^2
            # cross = S0 * sh_ss (since S0 = num/denom = dot(sl,sh)/sh_ss)
            # RSS = sl_ss - S0^2 * sh_ss = sl_ss - num^2/sh_ss
            rss     = (sl_ss[:, xp.newaxis]
                       - num ** 2 / (sh_ss[xp.newaxis, :] + 1e-30))
            rss     = xp.clip(rss, 0, None)
            ll_flat = -0.5 * (nTE * xp.log(xp.array(2*np.pi*sigma2, dtype=xp.float32))
                              + rss / sigma2)
        else:
            # Rician LL — computed without materialising (B, nPR, nTE).
            #
            # log p(s|nu,sigma) = log(s) - log(sigma^2) - (s^2+nu^2)/(2*sigma^2)
            #                     + log_I0(s*nu/sigma^2)
            #
            # Sum over echoes:
            #   LL = sum_t log(s_t)            [signal-only term, per voxel]
            #      - nTE*log(sigma^2)           [constant]
            #      - (||s||^2 + ||nu||^2) / (2*sigma^2)   [quadratic term]
            #      + sum_t log_I0(s_t * nu_t / sigma^2)   [Bessel term]
            #
            # nu_t = S0 * shape_t, so:
            #   ||nu||^2 = S0^2 * ||shape||^2  = S0^2 * denom   (nV, nPR)
            #   s_t*nu_t = s_t * S0 * shape_t  => sum_t = S0 * dot(s, shape) = S0 * num
            #
            # Quadratic terms: (B, nPR) — no (B, nPR, nTE) needed
            sl_ss    = xp.einsum('bi,bi->b', sl_b, sl_b)             # (B,)
            nu_ss    = S0**2 * denom[xp.newaxis, :]                   # (B, nPR)  ||nu||^2
            quad     = (sl_ss[:, xp.newaxis] + nu_ss) / (2 * sigma2) # (B, nPR)

            # Signal log-sum: sum_t log(s_t) — scalar per voxel
            log_s_sum = xp.sum(xp.log(xp.clip(sl_b, 1e-30, None)),
                               axis=1)                                 # (B,)

            # Bessel term: sum_t log_I0(s_t * S0 * shape_t / sigma^2)
            # Argument:  a_t = S0 * s_t * shape_t / sigma^2
            # sum_t log_I0(a_t) — we can't avoid the (B, nPR, nTE) here,
            # but we process it in grid chunks to cap memory.
            try:
                import psutil
                avail = psutil.virtual_memory().available
            except ImportError:
                avail = 1 * 1024**3
            # Each chunk needs (B, gr_chunk, nTE) × 2 arrays (scp, lb0) × 4 bytes
            gr_chunk = max(1, int(avail * 0.20 / (2 * B * nTE * 4)))
            gr_chunk = min(gr_chunk, nPR)

            bessel_sum = xp.zeros((B, nPR), dtype=xp.float32)
            for gr_start in range(0, nPR, gr_chunk):
                gr_end = min(gr_start + gr_chunk, nPR)
                S0_g   = S0[:, gr_start:gr_end]                       # (B, G)
                sh_g   = sh_flat[gr_start:gr_end, :]                  # (G, nTE)
                # scp[b,g,t] = S0[b,g] * sl[b,t] * sh[g,t] / sigma2
                scp    = xp.clip(
                    S0_g[:, :, xp.newaxis]
                    * sl_b[:, xp.newaxis, :]
                    * sh_g[xp.newaxis, :, :] / sigma2, 0, None)       # (B, G, nTE)
                if xp is np:
                    lb0 = _log_besseli0(scp)
                else:
                    lb0 = xp.where(scp < 3.75,
                                   scp**2/4.0 - xp.log(xp.array(2.0, dtype=xp.float32)),
                                   scp - 0.5*xp.log(2*np.pi*scp + 1e-30))
                bessel_sum[:, gr_start:gr_end] = xp.sum(lb0, axis=2) # (B, G)

            ll_flat = (log_s_sum[:, xp.newaxis]
                       - nTE * xp.log(xp.array(sigma2, dtype=xp.float32))
                       - quad
                       + bessel_sum)                                   # (B, nPR)

        # Mask background
        ll_flat = xp.where(msk_gpu[:, xp.newaxis],
                           ll_flat, xp.full_like(ll_flat, -xp.inf))

        # Best per voxel
        vox_idx  = xp.arange(B)
        best_idx = xp.argmax(ll_flat, axis=1)
        best_pi  = best_idx // n_r2
        best_ri  = best_idx %  n_r2

        b_pdff = pdff_arr[best_pi]
        b_r2   = r2_arr[best_ri]
        b_S0   = S0[vox_idx, best_idx]
        b_ll   = ll_flat[vox_idx, best_idx]

        # Second-best (PDFF >0.15 away)
        far_mask = (xp.abs(pdff_arr[xp.newaxis, :] - b_pdff[:, xp.newaxis]) > 0.15)
        far_flat = xp.repeat(far_mask, n_r2, axis=1)
        ll_far   = xp.where(far_flat, ll_flat, xp.full_like(ll_flat, -xp.inf))
        sec_idx  = xp.argmax(ll_far, axis=1)
        sec_pi   = sec_idx // n_r2
        sec_ri   = sec_idx %  n_r2
        s_pdff   = pdff_arr[sec_pi]
        s_r2     = r2_arr[sec_ri]
        s_ll     = ll_far[vox_idx, sec_idx]

        # Copy back to CPU
        def to_cpu(a): return a.get() if xp is not np else a
        best_pdff_all[start:end] = to_cpu(b_pdff)
        best_r2_all[start:end]   = to_cpu(b_r2)
        best_S0_all[start:end]   = to_cpu(b_S0)
        best_ll_all[start:end]   = to_cpu(b_ll)
        sec_pdff_all[start:end]  = to_cpu(s_pdff)
        sec_r2_all[start:end]    = to_cpu(s_r2)
        sec_ll_all[start:end]    = to_cpu(s_ll)

    return dict(
        PDFF      = best_pdff_all.reshape(nX, nY),
        R2        = best_r2_all.reshape(nX, nY),
        S0        = best_S0_all.reshape(nX, nY),
        loglik    = best_ll_all.reshape(nX, nY),
        PDFF_opt2 = sec_pdff_all.reshape(nX, nY),
        R2_opt2   = sec_r2_all.reshape(nX, nY),
        loglik_opt2 = sec_ll_all.reshape(nX, nY),
    )


def fit_image(img_4d: np.ndarray,
              TE_ms: np.ndarray,
              tesla: float,
              sigma: float,
              slice_idx: int = 0,
              indent: int = 0,
              methods: tuple = ('gaussian', 'rician'),
              verbose: bool = True,
              _precomputed=None) -> dict:
    """
    Fit all voxels in a single slice using fully vectorised operations.

    Automatically uses GPU (via CuPy) when available — no code changes needed.

    Parameters
    ----------
    img_4d       : (nX, nY, nZ, nTE) magnitude image array
    TE_ms        : (nTE,) echo times in milliseconds
    tesla        : field strength in Tesla
    sigma        : noise SD estimate
    slice_idx    : which z-slice to fit
    indent       : border voxels to skip
    methods      : subset of ('gaussian', 'rician')
    verbose      : print progress
    _precomputed : optional (shapes, pdff_grid, r2_grid) tuple to avoid
                   recomputing the grid when fitting multiple slices

    Returns
    -------
    maps : dict with PDFF (0-1), R2 (ms^-1), S0, loglik, likDiff maps
    """
    nX, nY = img_4d.shape[0], img_4d.shape[1]

    if _precomputed is not None:
        shapes, pdff_grid, r2_grid = _precomputed
    else:
        if verbose:
            print("  Pre-computing signal shapes …")
        shapes, pdff_grid, r2_grid = precompute_signal_shapes(TE_ms, tesla)

    slice_mag = img_4d[:, :, slice_idx, :]   # (nX, nY, nTE)

    # Background mask: voxels with signal > 50% of sigma
    mask = (np.max(slice_mag, axis=2) >= sigma * 0.5).astype(np.float32)
    if indent > 0:
        mask[:indent, :]  = 0;  mask[-indent:, :] = 0
        mask[:, :indent]  = 0;  mask[:, -indent:] = 0

    if verbose:
        xp_name = 'GPU (CuPy)' if _GPU_AVAILABLE else 'CPU (NumPy)'
        n_tissue = int(mask.sum())
        print(f"  Backend: {xp_name}  |  tissue voxels: {n_tissue}/{nX*nY}")

    t0   = time.time()
    maps = {}

    for m in methods:
        result = _fit_slice_vectorised(
            slice_mag, shapes, pdff_grid, r2_grid, sigma, m, mask, _xp)

        maps[f'PDFF_{m}']      = result['PDFF']
        maps[f'R2_{m}']        = result['R2']
        maps[f'S0_{m}']        = result['S0']
        maps[f'loglik_{m}']    = result['loglik']
        maps[f'PDFF_{m}_opt2'] = result['PDFF_opt2']
        maps[f'R2_{m}_opt2']   = result['R2_opt2']

        if verbose:
            print(f"  {m.capitalize()} fit done in {time.time()-t0:.1f}s")
        t0 = time.time()

    # MAGORINO post-processing: spatially filtered likelihood map
    if 'rician' in methods:
        lld = maps['loglik_rician']
        maps['likDiff_gauss3'] = gaussian_filter(lld, sigma=1.5)
        maps['likDiff_gauss5'] = gaussian_filter(lld, sigma=2.5)
        maps['likDiff_box3']   = uniform_filter(lld, size=3)
        maps['likDiff_box5']   = uniform_filter(lld, size=5)

    return maps


# ---------------------------------------------------------------------------
# 6.  Helper: convert R2* units and compute T2*
# ---------------------------------------------------------------------------

def r2star_ms_to_s(R2_map_ms: np.ndarray) -> np.ndarray:
    """Convert R2* from ms^-1 to s^-1 (multiply by 1000)."""
    return R2_map_ms * 1000.0


def t2star_ms(R2_map_ms: np.ndarray) -> np.ndarray:
    """Compute T2* in ms from R2* in ms^-1."""
    with np.errstate(divide='ignore', invalid='ignore'):
        t2s = np.where(R2_map_ms > 0, 1.0 / R2_map_ms, 0.0)
    return t2s


# ---------------------------------------------------------------------------
# 7.  Save output maps as NIfTI
# ---------------------------------------------------------------------------

def save_maps_nifti(maps: dict,
                    ref_img: nib.Nifti1Image,
                    out_dir: Path,
                    slice_idx: int):
    """
    Save each map array as a NIfTI file, with the affine/qform/sform
    correctly offset to the z-position of slice_idx in the original volume.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_hdr = ref_img.header
    vol_affine = ref_img.affine.copy()

    # Shift the z-origin to the world position of this specific slice.
    # The 4th column of the affine is the origin [x0, y0, z0, 1].
    # For a diagonal affine (no oblique), z_world = z0 + slice_idx * dz.
    # For an oblique affine the correct approach is to transform the
    # voxel coordinate [0, 0, slice_idx] to world space and use that z.
    slice_affine = vol_affine.copy()
    # World position of voxel (0, 0, slice_idx)
    origin_vox   = np.array([0.0, 0.0, float(slice_idx), 1.0])
    origin_world = vol_affine @ origin_vox          # (x_w, y_w, z_w, 1)
    slice_affine[:3, 3] = origin_world[:3]

    # Names of maps that are stored internally as fractions (0-1) but
    # should be saved as percentages (0-100)
    _PDFF_KEYS = {'PDFF_gaussian', 'PDFF_rician',
                  'PDFF_gaussian_opt2', 'PDFF_rician_opt2'}

    for name, data in maps.items():
        if not isinstance(data, np.ndarray):
            continue
        save_data = data.astype(np.float32)
        if name in _PDFF_KEYS:
            save_data = (save_data * 100.0)   # 0-1 → 0-100 %

        new_img = nib.Nifti1Image(save_data, slice_affine)
        new_img.header.set_qform(slice_affine, code=int(ref_hdr['qform_code']))
        new_img.header.set_sform(slice_affine, code=int(ref_hdr['sform_code']))
        new_img.header['pixdim'][:4] = ref_hdr['pixdim'][:4]

        fname = out_dir / f"{name}_slice{slice_idx:02d}.nii.gz"
        nib.save(new_img, str(fname))
        print(f"  Saved: {fname}")


class Magorino(Module):
    def __init__(self, name='magorino', **kwargs):
        Module.__init__(self, name, **kwargs)

    def process(self):
        t2star_dir = self.kwargs.get('t2star_dir', 't2star')
        echos_glob = self.kwargs.get('echos_glob', 't2star_e_*.nii.gz')
        expected_echos = self.kwargs.get('expected_echos', None)
        field_strength = float(self.kwargs.get('field_strength', 3.0))
        indent = int(self.kwargs.get('indent', 0))
        sigma = self.kwargs.get('sigma', None)
        method = self.kwargs.get('method', 'all')
        roi_x = self.kwargs.get('roi_x', (10, 30))
        roi_y = self.kwargs.get('roi_y', (10, 30))
        roi_z = self.kwargs.get('roi_z', None)

        echos = self.inimgs(t2star_dir, echos_glob)
        if not echos:
            self.no_data('No MAGORINO multi-echo data found')
        elif expected_echos and len(echos) != expected_echos:
            self.bad_data(f'Expected {expected_echos} echos, got {len(echos)}')

        echos.sort(key=lambda img: img.EchoTime)
        imgdata = [echo.data for echo in echos]
        tes = np.array([echo.EchoTime for echo in echos], dtype=np.float64)
        if np.all(tes < 1):
            LOG.info(' - Looks like TEs were specified in seconds - converting to ms')
            tes = tes * 1000.0

        src = echos[-1]
        data = np.stack(imgdata, axis=-1).astype(np.float64)
        methods = ('gaussian', 'rician') if method == 'all' else (method,)
        if any(m not in ('gaussian', 'rician') for m in methods):
            self.bad_data(f'Unsupported MAGORINO method selection: {method}')

        LOG.info(f' - Found {len(echos)} echoes for MAGORINO fit')
        LOG.info(f' - TEs (ms): {tes}')
        LOG.info(f' - Methods: {methods}')
        LOG.info(f' - Field strength: {field_strength} T')
        LOG.info(f' - GPU available: {_GPU_AVAILABLE}')
        src.save_derived(src.data, self.outfile('last_echo.nii.gz'))

        sigma = self._resolve_sigma(data, tes, field_strength, sigma, roi_x, roi_y, roi_z)
        LOG.info(f' - Using sigma={sigma:.3f}')

        shapes, pdff_grid, r2_grid = precompute_signal_shapes(tes, field_strength)
        all_maps = {}
        for slice_idx in range(data.shape[2]):
            LOG.info(f' - Fitting slice {slice_idx + 1}/{data.shape[2]}')
            all_maps[slice_idx] = fit_image(
                data,
                tes,
                field_strength,
                sigma,
                slice_idx=slice_idx,
                indent=indent,
                methods=methods,
                verbose=False,
                _precomputed=(shapes, pdff_grid, r2_grid),
            )

        self._save_volumes(src, all_maps, methods)

    def _resolve_sigma(self, data, tes, field_strength, sigma, roi_x, roi_y, roi_z):
        if sigma is not None:
            return float(sigma)

        x0, x1 = self._normalise_roi_range(roi_x)
        y0, y1 = self._normalise_roi_range(roi_y)
        if roi_z is None:
            mid = data.shape[2] // 2
            z0, z1 = max(0, mid - 2), min(data.shape[2], mid + 2)
        else:
            z0, z1 = self._normalise_roi_range(roi_z)

        roi_mag = np.abs(data[x0:x1, y0:y1, z0:z1, :])
        if roi_mag.size == 0:
            self.bad_data('Sigma estimation ROI is empty')
        roi_signal = np.median(roi_mag.reshape(-1, roi_mag.shape[-1]), axis=0)
        LOG.info(f' - Estimating sigma from ROI x={[x0, x1]}, y={[y0, y1]}, z={[z0, z1]}')
        return float(estimate_sigma_from_roi(tes, field_strength, roi_signal))

    def _normalise_roi_range(self, roi_range):
        if isinstance(roi_range, str):
            parts = [int(part) for part in roi_range.split(',')]
        else:
            parts = [int(part) for part in roi_range]
        if len(parts) != 2:
            self.bad_data(f'ROI range must have two values, got {roi_range}')
        return parts[0], parts[1]

    def _save_volumes(self, src, all_maps, methods):
        map_keys = [
            key for key, value in all_maps[0].items()
            if isinstance(value, np.ndarray)
        ]
        n_x, n_y = src.shape[:2]
        n_z = len(all_maps)
        pdff_keys = {
            'PDFF_gaussian', 'PDFF_rician',
            'PDFF_gaussian_opt2', 'PDFF_rician_opt2',
        }

        for key in map_keys:
            volume = np.zeros((n_x, n_y, n_z), dtype=np.float32)
            for slice_idx, maps in all_maps.items():
                volume[:, :, slice_idx] = maps[key].astype(np.float32)
            out_data = volume * 100.0 if key in pdff_keys else volume
            src.save_derived(out_data, self.outfile(f'{key.lower()}.nii.gz'))

        for fit_method in methods:
            r2_key = f'R2_{fit_method}'
            t2_data = np.zeros((n_x, n_y, n_z), dtype=np.float32)
            r2_data = np.zeros((n_x, n_y, n_z), dtype=np.float32)
            for slice_idx, maps in all_maps.items():
                r2_data[:, :, slice_idx] = r2star_ms_to_s(maps[r2_key]).astype(np.float32)
                t2_data[:, :, slice_idx] = t2star_ms(maps[r2_key]).astype(np.float32)
            src.save_derived(r2_data, self.outfile(f'r2star_{fit_method}.nii.gz'))
            src.save_derived(t2_data, self.outfile(f't2star_{fit_method}.nii.gz'))


# ---------------------------------------------------------------------------
# 8.  Visualisation
# ---------------------------------------------------------------------------

def plot_maps(maps: dict, slice_idx: int, sigma: float, out_path: Path = None):
    """Display PDFF and R2* maps for MAGO and MAGORINO."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    has_gaussian = 'PDFF_gaussian' in maps
    has_rician = 'PDFF_rician' in maps

    n_cols = (2 if has_gaussian else 0) + (2 if has_rician else 0) + (1 if has_gaussian and has_rician else 0)
    fig, axes = plt.subplots(2, max(n_cols, 1), figsize=(5 * n_cols, 9))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    col = 0
    if has_gaussian:
        pdff_g = np.clip(maps['PDFF_gaussian'] * 100.0, 0, 100)
        r2_g = r2star_ms_to_s(maps['R2_gaussian'])
        im0 = axes[0, col].imshow(pdff_g.T, cmap='viridis', vmin=0, vmax=100, origin='lower')
        axes[0, col].set_title(f'PDFF (%) - MAGO (Gaussian)\nslice {slice_idx}')
        plt.colorbar(im0, ax=axes[0, col], label='%')
        im1 = axes[1, col].imshow(r2_g.T, cmap='hot', vmin=0, vmax=200, origin='lower')
        axes[1, col].set_title('R2* [s^-1] - MAGO')
        plt.colorbar(im1, ax=axes[1, col])
        col += 1

    if has_rician:
        pdff_r = np.clip(maps['PDFF_rician'] * 100.0, 0, 100)
        r2_r = r2star_ms_to_s(maps['R2_rician'])
        im2 = axes[0, col].imshow(pdff_r.T, cmap='viridis', vmin=0, vmax=100, origin='lower')
        axes[0, col].set_title(f'PDFF (%) - MAGORINO (Rician)\nslice {slice_idx}')
        plt.colorbar(im2, ax=axes[0, col], label='%')
        im3 = axes[1, col].imshow(r2_r.T, cmap='hot', vmin=0, vmax=200, origin='lower')
        axes[1, col].set_title('R2* [s^-1] - MAGORINO')
        plt.colorbar(im3, ax=axes[1, col])
        col += 1

    if has_gaussian and has_rician:
        diff_pdff = (maps['PDFF_rician'] - maps['PDFF_gaussian']) * 100.0
        diff_r2 = r2star_ms_to_s(maps['R2_rician'] - maps['R2_gaussian'])
        im4 = axes[0, col].imshow(diff_pdff.T, cmap='bwr', vmin=-10, vmax=10, origin='lower')
        axes[0, col].set_title('Delta PDFF % (Rician - Gaussian)')
        plt.colorbar(im4, ax=axes[0, col], label='%')
        im5 = axes[1, col].imshow(diff_r2.T, cmap='bwr', vmin=-50, vmax=50, origin='lower')
        axes[1, col].set_title('Delta R2* [s^-1] (Rician - Gaussian)')
        plt.colorbar(im5, ax=axes[1, col])

    plt.suptitle(f'MAGO / MAGORINO fit | sigma = {sigma:.1f} | slice {slice_idx}', fontsize=13, y=1.01)
    plt.tight_layout()

    if out_path is not None:
        plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9.  Main entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MAGO / MAGORINO R2*+PDFF fitting',
        formatter_class=argparse.RawTextHelpFormatter)

    # Two mutually exclusive input modes:
    #   1. Single 4-D NIfTI  (old behaviour)
    #   2. Folder of per-echo 3-D NIfTIs named *_e1.nii.gz ... *_eN.nii.gz
    input_grp = parser.add_mutually_exclusive_group(required=False)
    input_grp.add_argument(
        '--input',
        help='Single 4-D multi-echo NIfTI file (.nii.gz).\n'
             'The 4th dimension must be echo number.')
    input_grp.add_argument(
        '--input-dir',
        dest='input_dir',
        help='Folder containing per-echo 3-D NIfTI files named\n'
             '  <prefix>_e1.nii.gz  <prefix>_e2.nii.gz  …  <prefix>_eN.nii.gz\n'
             'All files matching *_e*.nii.gz in that folder are loaded and\n'
             'stacked in echo order.')

    parser.add_argument('--output',  default='output_maps',
                        help='Directory for output NIfTI maps and figures (default: output_maps)')
    parser.add_argument('--slice',   type=str, default='all',
                        help='Slice index to fit (0-based), or "all" (default: all)')
    parser.add_argument('--method',  default='all',
                        choices=['all', 'gaussian', 'rician'],
                        help='Which fitting method(s) to run')
    parser.add_argument('--indent',  type=int, default=0,
                        help='Border voxels to skip (default 0)')
    parser.add_argument('--field',   type=float, default=3.0,
                        help='Field strength in Tesla (default 3.0)')
    parser.add_argument('--TE',      type=str, default=None,
                        help='Echo times in ms, comma-separated, e.g. "1.45,2.91,4.37".\n'
                             'If omitted, the script will attempt to read TE values from\n'
                             'the NIfTI header pixdim or prompt you to supply them.')
    parser.add_argument('--sigma',   type=float, default=None,
                        help='Noise sigma (if None, estimated from data)')
    parser.add_argument('--roi-x',   type=str, default='10,30',
                        help='ROI x range for sigma estimation, e.g. "10,30"')
    parser.add_argument('--roi-y',   type=str, default='10,30',
                        help='ROI y range for sigma estimation, e.g. "10,30"')
    parser.add_argument('--roi-z',   type=str, default=None,
                        help='ROI z (slice) range for sigma estimation, e.g. "18,22".\n'
                             'Defaults to the middle 4 slices of the volume.')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Load data — either from a single 4-D file or a folder of per-echo files
    # -----------------------------------------------------------------------
    import re

    if args.input_dir is not None:
        # --- Per-echo folder mode ---
        echo_dir = Path(args.input_dir)
        echo_files = sorted(
            echo_dir.glob('*_e*.nii.gz'),
            key=lambda p: int(re.search(r'_e(\d+)\.nii\.gz$', p.name).group(1)))
        if not echo_files:
            raise FileNotFoundError(
                f"No files matching *_e*.nii.gz found in {echo_dir}")
        print(f"\n[1/5] Loading {len(echo_files)} per-echo files from: {echo_dir}")
        for f in echo_files:
            print(f"      {f.name}")
        echo_vols = [nib.load(str(f)).get_fdata().astype(np.float64)
                     for f in echo_files]
        img = np.stack(echo_vols, axis=-1)   # (nX, nY, nZ, nTE)
        nii = nib.load(str(echo_files[0]))   # use first echo for affine/header
        nTE_data = len(echo_files)
    else:
        # --- Single 4-D file mode ---
        input_path = args.input if args.input else '/tmp/t2star_all.nii.gz'
        print(f"\n[1/5] Loading image: {input_path}")
        nii = nib.load(input_path)
        img = nii.get_fdata().astype(np.float64)
        nTE_data = img.shape[3] if img.ndim == 4 else 1

    nZ = img.shape[2]
    print(f"      Shape: {img.shape}  (nX, nY, nZ, nTE)")
    print(f"      Number of echoes detected: {nTE_data}")

    # -----------------------------------------------------------------------
    # Resolve echo times
    # -----------------------------------------------------------------------
    if args.TE is not None:
        # User supplied TEs explicitly
        TE_ms = np.array([float(x) for x in args.TE.split(',')])
    else:
        # Try to infer from header: pixdim[4] is often the echo spacing in s
        te_spacing_s = float(nii.header['pixdim'][4])
        if te_spacing_s > 0 and te_spacing_s < 1.0:
            # Looks like a genuine TE spacing in seconds
            TE_ms = np.arange(1, nTE_data + 1) * te_spacing_s * 1000.0
            print(f"      TE spacing from header pixdim[4]: {te_spacing_s*1000:.4f} ms")
        else:
            raise ValueError(
                f"Could not determine echo times automatically "
                f"(pixdim[4]={te_spacing_s:.4f} s does not look like a TE spacing).\n"
                f"Please supply them explicitly with --TE, e.g.:\n"
                f"  --TE 1.45,2.91,4.37,5.83,...  (one value per echo, in ms)")

    # Validate: number of TEs must match number of echoes in data
    if len(TE_ms) != nTE_data:
        raise ValueError(
            f"Number of TE values ({len(TE_ms)}) does not match "
            f"number of echoes in data ({nTE_data}).\n"
            f"Please check --TE or your input files.")

    tesla   = args.field
    methods = ('gaussian', 'rician') if args.method == 'all' else (args.method,)

    # Resolve which slices to fit
    if args.slice.lower() == 'all':
        slices_to_fit = list(range(nZ))
    else:
        slices_to_fit = [int(args.slice)]

    print(f"      Echo times (ms): {TE_ms}")
    print(f"      Field strength:  {tesla} T")
    print(f"      Fitting methods: {methods}")
    print(f"      Slices:          {slices_to_fit if len(slices_to_fit) <= 5 else f'0-{nZ-1} ({nZ} slices)'}")

    # -----------------------------------------------------------------------
    # Sigma estimation (done once using a mid-volume ROI)
    # -----------------------------------------------------------------------
    print("\n[2/5] Estimating noise sigma …")

    if args.sigma is not None:
        sigma = args.sigma
        print(f"      Using user-supplied sigma = {sigma:.2f}")
    else:
        rx = [int(x) for x in args.roi_x.split(',')]
        ry = [int(x) for x in args.roi_y.split(',')]
        if args.roi_z is not None:
            rz = [int(x) for x in args.roi_z.split(',')]
        else:
            mid = nZ // 2
            rz  = [max(0, mid - 2), min(nZ, mid + 2)]
        roi_mag    = np.abs(img[rx[0]:rx[1], ry[0]:ry[1], rz[0]:rz[1], :])
        roi_signal = np.median(roi_mag.reshape(-1, roi_mag.shape[-1]), axis=0)
        sigma = estimate_sigma_from_roi(TE_ms, tesla, roi_signal)
        print(f"      ROI x={rx}, y={ry}, z={rz}")
        print(f"      Estimated sigma = {sigma:.2f}")

    # -----------------------------------------------------------------------
    # Pre-compute signal shapes once for all slices
    # -----------------------------------------------------------------------
    print("\n[3/5] Pre-computing signal shapes …")
    shapes, pdff_grid, r2_grid = precompute_signal_shapes(TE_ms, tesla)

    # -----------------------------------------------------------------------
    # Fit slices
    # -----------------------------------------------------------------------
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_maps = {}
    for sl_idx, slice_idx in enumerate(slices_to_fit):
        print(f"\n[4/5] Fitting slice {slice_idx}  ({sl_idx+1}/{len(slices_to_fit)}) …")
        maps = fit_image(img, TE_ms, tesla, sigma,
                         slice_idx=slice_idx,
                         indent=args.indent,
                         methods=methods,
                         verbose=True,
                         _precomputed=(shapes, pdff_grid, r2_grid))
        all_maps[slice_idx] = maps

        if len(slices_to_fit) == 1:
            # Single slice — save immediately and produce figure
            save_maps_nifti(maps, nii, out_dir, slice_idx)
            origin_world = nii.affine @ np.array([0.0, 0.0, float(slice_idx), 1.0])
            slice_affine = nii.affine.copy()
            slice_affine[:3, 3] = origin_world[:3]
            for m in methods:
                for map_name, data in [
                        (f'R2star_per_s_{m}', r2star_ms_to_s(maps[f'R2_{m}'])),
                        (f'T2star_ms_{m}',    t2star_ms(maps[f'R2_{m}']))]:
                    out_img = nib.Nifti1Image(data.astype(np.float32), slice_affine)
                    out_img.header.set_qform(slice_affine, code=int(nii.header['qform_code']))
                    out_img.header.set_sform(slice_affine, code=int(nii.header['sform_code']))
                    out_img.header['pixdim'][:4] = nii.header['pixdim'][:4]
                    nib.save(out_img, str(out_dir / f'{map_name}_slice{slice_idx:02d}.nii.gz'))
            fig_path = out_dir / f'maps_slice{slice_idx:02d}.png'
            plot_maps(maps, slice_idx, sigma, out_path=fig_path)

    # -----------------------------------------------------------------------
    # If fitting all slices, also assemble 3-D NIfTI volumes
    # -----------------------------------------------------------------------
    _PDFF_KEYS = {'PDFF_gaussian', 'PDFF_rician',
                  'PDFF_gaussian_opt2', 'PDFF_rician_opt2'}

    def _save_vol(vol_data, name, affine, header):
        out_img = nib.Nifti1Image(vol_data.astype(np.float32), affine)
        out_img.header.set_qform(affine, code=int(header['qform_code']))
        out_img.header.set_sform(affine, code=int(header['sform_code']))
        out_img.header['pixdim'][:4] = header['pixdim'][:4]
        nib.save(out_img, str(out_dir / f'{name}_volume.nii.gz'))
        print(f"  Saved 3-D volume: {name}_volume.nii.gz")

    if len(slices_to_fit) > 1:
        print(f"\n[5/5] Assembling 3-D output volumes …")
        nX, nY = img.shape[0], img.shape[1]
        map_keys = [k for k in all_maps[slices_to_fit[0]].keys()
                    if isinstance(all_maps[slices_to_fit[0]][k], np.ndarray)]

        for key in map_keys:
            vol = np.zeros((nX, nY, nZ), dtype=np.float32)
            for sl in slices_to_fit:
                vol[:, :, sl] = all_maps[sl][key].astype(np.float32)
            if key in _PDFF_KEYS:
                vol = vol * 100.0
            _save_vol(vol, key, nii.affine, nii.header)

        for m in methods:
            r2_vol = np.zeros((nX, nY, nZ), dtype=np.float32)
            t2_vol = np.zeros((nX, nY, nZ), dtype=np.float32)
            for sl in slices_to_fit:
                r2_vol[:, :, sl] = r2star_ms_to_s(all_maps[sl][f'R2_{m}']).astype(np.float32)
                t2_vol[:, :, sl] = t2star_ms(all_maps[sl][f'R2_{m}']).astype(np.float32)
            _save_vol(r2_vol, f'R2star_per_s_{m}', nii.affine, nii.header)
            _save_vol(t2_vol, f'T2star_ms_{m}',    nii.affine, nii.header)

        # Save a figure for the middle fitted slice
        mid_sl   = slices_to_fit[len(slices_to_fit) // 2]
        fig_path = out_dir / f'maps_slice{mid_sl:02d}.png'
        plot_maps(all_maps[mid_sl], mid_sl, sigma, out_path=fig_path)
        print(f"  Figure saved for middle slice ({mid_sl}): {fig_path}")
    else:
        print("\n[5/5] Single slice — skipping 3-D assembly.")

    print("\nDone.")
    return all_maps, sigma


if __name__ == '__main__':
    main()