# diagnostics.py
"""
Diagnostic computations for PEC analysis.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from constants import MIN_STD_SAMPLES, STD_DDOF


DEFAULT_DIAGNOSTICS = {
    "bias_gamma": False,
    "variance_decomp": False,
    "bias_analytical": False,
    "weight_variance": False,
    "baseline_tracking": False,
}


def estimate_bias_from_samples(
    measurements: np.ndarray,
    weights: np.ndarray,
    w0: int,
) -> Tuple[float, float, float]:
    """Estimate bias using γ_diff from paper."""
    mask_in = weights <= w0
    mask_out = weights > w0
    if mask_in.sum() < MIN_STD_SAMPLES or mask_out.sum() < MIN_STD_SAMPLES:
        return float("nan"), float("nan"), float("nan")

    o_in = float(measurements[mask_in].mean())
    o_out = float(measurements[mask_out].mean())
    if abs(o_in) < 1e-10 or abs(o_out) < 1e-10:
        return float("nan"), float("nan"), float("nan")

    gamma_diff = o_out / o_in
    bias_bound = abs(gamma_diff**2 - 1.0)

    sigma_in = float(measurements[mask_in].std(ddof=STD_DDOF)) / np.sqrt(mask_in.sum())
    sigma_out = float(measurements[mask_out].std(ddof=STD_DDOF)) / np.sqrt(mask_out.sum())
    gamma_std = abs(gamma_diff) * np.sqrt((sigma_in / o_in) ** 2 + (sigma_out / o_out) ** 2)

    return bias_bound, gamma_diff, gamma_std


def decompose_variance(raw_estimates: np.ndarray) -> Tuple[float, float, float]:
    """Decompose variance into circuit/shot components."""
    if raw_estimates.size < MIN_STD_SAMPLES:
        return 0.0, 0.0, 0.0
    var_total = float(raw_estimates.var(ddof=STD_DDOF))
    return var_total, max(0.0, var_total), 0.0


def analytical_bias_bound(error_locs: List[Tuple], beta_thresh: float) -> float:
    """Compute analytical upper bound on bias."""
    if beta_thresh <= 0.0:
        return 0.0
    suppression = 1.0 - float(np.exp(-beta_thresh))
    total_error = 0.0
    for (_, _, p) in error_locs:
        p_error = 1.0 - float(p[0])
        total_error += suppression * p_error
    return 2.0 * total_error


def weight_conditioned_variance(
    estimates: np.ndarray,
    weights: np.ndarray,
    importance_weights: np.ndarray,
) -> dict:
    """Compute variance conditioned on Pauli weight."""
    weight_vals = weights.astype(int)
    unique_weights = np.unique(weight_vals)
    by_weight: Dict[int, dict] = {}
    for w in unique_weights:
        mask = weight_vals == w
        if mask.sum() < MIN_STD_SAMPLES:
            continue
        w_estimates = estimates[mask]
        w_iw = importance_weights[mask]
        var_est = float(w_estimates.var(ddof=STD_DDOF))
        by_weight[int(w)] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "mean_estimate": float(w_estimates.mean()),
            "var_estimate": var_est,
            "mean_iw": float(w_iw.mean()),
            "var_iw": float(w_iw.var(ddof=STD_DDOF)),
            "var_contribution": float(mask.mean()) * var_est,
        }

    total_var = float(estimates.var(ddof=STD_DDOF)) if estimates.size >= MIN_STD_SAMPLES else 0.0
    high_weight_var = sum(stats["var_contribution"] for w, stats in by_weight.items() if w > 2)
    return {
        "by_weight": by_weight,
        "total_var": total_var,
        "high_weight_fraction": high_weight_var / total_var if total_var > 0.0 else 0.0,
    }


def baseline_tracking(
    estimates: np.ndarray,
    weights: np.ndarray,
    batch_indices: Optional[np.ndarray],
) -> Tuple[float, float, int, bool, float]:
    """Track noisy baseline for drift detection."""
    mask_w0 = weights == 0
    count = int(mask_w0.sum())
    if count == 0:
        return float("nan"), float("nan"), 0, False, float("nan")

    mean = float(estimates[mask_w0].mean())
    std = (
        float(estimates[mask_w0].std(ddof=STD_DDOF)) / np.sqrt(count)
        if count >= MIN_STD_SAMPLES
        else 0.0
    )

    if batch_indices is None:
        return mean, std, count, False, float("nan")

    batches = np.unique(batch_indices)
    if batches.size < 2:
        return mean, std, count, False, float("nan")

    batch_means: List[float] = []
    batch_stds: List[float] = []
    for b in batches:
        batch_mask = mask_w0 & (batch_indices == b)
        if batch_mask.sum() < 1:
            continue
        batch_means.append(float(estimates[batch_mask].mean()))
        if batch_mask.sum() >= MIN_STD_SAMPLES:
            batch_std = float(estimates[batch_mask].std(ddof=STD_DDOF)) / np.sqrt(batch_mask.sum())
        else:
            batch_std = 0.0
        batch_stds.append(batch_std)

    if len(batch_means) < 2:
        return mean, std, count, False, float("nan")

    denom = np.sqrt(batch_stds[0] ** 2 + batch_stds[-1] ** 2)
    z_score = abs(batch_means[-1] - batch_means[0]) / denom if denom > 0 else float("inf")
    drift = z_score > 2.0
    return mean, std, count, drift, float(z_score)