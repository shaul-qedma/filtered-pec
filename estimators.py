# estimators.py
"""
PEC estimator result types.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PECEstimate:
    """Result of standard PEC estimation."""
    mean: float
    std: float
    qp_norm: float
    n_samples: int


@dataclass
class ThresholdPECEstimate:
    """Result of threshold-filtered PEC estimation."""
    mean: float
    std: float
    qp_norm_proposal: float
    effective_samples: float
    n_samples: int
    weight_mean: float
    weight_std: float
    weight_above_frac: float
    # Optional diagnostics
    bias_bound_gamma: float = float("nan")
    gamma_diff: float = float("nan")
    gamma_std: float = float("nan")
    bias_bound_analytical: float = float("nan")
    var_total: float = float("nan")
    var_circuit: float = float("nan")
    var_shot: float = float("nan")
    weight_variance: Optional[dict] = None
    baseline_mean: float = float("nan")
    baseline_std: float = float("nan")
    baseline_count: int = 0
    baseline_drift: bool = False
    baseline_z: float = float("nan")