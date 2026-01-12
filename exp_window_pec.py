# exp_window_pec.py
"""
Exponential Window PEC Sampler
"""

from __future__ import annotations

from typing import List, Tuple
import sys

import numpy as np

from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from backend import Backend, QiskitStatevector
from constants import DEFAULT_BATCH_SIZE
from estimators import PECEstimate
from pec_shared import ETA, Circuit

console = Console()

VERIFY_TEST_CASES = (
    ("Depolarizing 5%", np.array([0.95, 0.05 / 3, 0.05 / 3, 0.05 / 3])),
    ("Depolarizing 10%", np.array([0.90, 0.10 / 3, 0.10 / 3, 0.10 / 3])),
    ("Asymmetric", np.array([0.90, 0.05, 0.03, 0.02])),
    ("Z-dominated", np.array([0.85, 0.02, 0.03, 0.10])),
)


def exp_window_quasi_prob(p: np.ndarray, beta: float) -> np.ndarray:
    """Compute local quasi-probability with exponential window."""
    eigenvalues = ETA @ p
    decay = np.exp(-beta)
    h = np.array([
        1.0 / eigenvalues[0],
        decay / eigenvalues[1],
        decay / eigenvalues[2],
        decay / eigenvalues[3],
    ])
    return 0.25 * (ETA @ h)


def qp_norm(q: np.ndarray) -> float:
    """Sampling overhead γ = ||q||₁"""
    return float(np.abs(q).sum())


def pec_estimate(
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    noise: dict,
    error_locs: List[Tuple],
    beta: float,
    n_samples: int,
    seed: int = 0,
    backend: Backend | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: bool = False,
) -> PECEstimate:
    """
    PEC estimation with exponential window filter.
    
    Args:
        circuit: Quantum circuit
        observable: Pauli string (e.g., "XZI")
        initial_state: Initial state vector
        noise: Noise model dict
        error_locs: List of (layer, qubit, noise_probs) tuples
        beta: Exponential suppression parameter (0 = full PEC)
        n_samples: Number of Monte Carlo samples
        seed: Random seed
        backend: Simulation backend (default: QiskitStatevector)
        batch_size: Batch size for circuit execution
    
    Returns:
        PECEstimate with mean, std, qp_norm, n_samples
    """
    if backend is None:
        backend = QiskitStatevector(batch_size=batch_size)
    
    rng = np.random.default_rng(seed)
    
    # Compute local quasi-probabilities
    local_q = [exp_window_quasi_prob(p, beta) for (_, _, p) in error_locs]
    local_qp_norm = [qp_norm(q) for q in local_q]
    total_qp_norm = float(np.prod(local_qp_norm))
    
    sampling_probs = [np.abs(q) / g for q, g in zip(local_q, local_qp_norm)]
    sampling_signs = [np.sign(q) for q in local_q]
    
    estimates = np.empty(n_samples)
    effective_batch = n_samples if batch_size <= 0 else batch_size
    
    show_progress = progress and sys.stderr.isatty()
    sample_bar = None
    if show_progress:
        sample_bar = tqdm(
            total=n_samples,
            desc="Samples",
            unit="sample",
            dynamic_ncols=True,
            leave=False,
            disable=not show_progress,
        )

    for start in range(0, n_samples, effective_batch):
        end = min(n_samples, start + effective_batch)
        
        insertions_list = []
        signs = np.empty(end - start)
        
        for i in range(start, end):
            insertions = {}
            sign = 1.0
            for v, (layer, qubit, _) in enumerate(error_locs):
                s = rng.choice(4, p=sampling_probs[v])
                insertions[(layer, qubit)] = s
                sign *= sampling_signs[v][s]
            insertions_list.append(insertions)
            signs[i - start] = sign
        
        measurements = backend.batch_expectations(
            circuit, observable, initial_state, noise, insertions_list
        )
        
        for j, measurement in enumerate(measurements):
            estimates[start + j] = signs[j] * measurement
        if sample_bar is not None:
            sample_bar.update(end - start)
    if sample_bar is not None:
        sample_bar.close()
    
    mean = total_qp_norm * float(estimates.mean())
    std = total_qp_norm * float(estimates.std()) / np.sqrt(n_samples)
    
    return PECEstimate(mean=mean, std=std, qp_norm=total_qp_norm, n_samples=n_samples)


def verify_beta_zero_is_full_pec():
    """Verify that β=0 recovers full PEC quasi-probabilities."""
    console.rule("Verification: β=0 equals Full PEC")
    table = Table(box=box.ASCII)
    table.add_column("Case")
    table.add_column("max|diff|", justify="right")
    table.add_column("Status", justify="center")

    for name, p in VERIFY_TEST_CASES:
        eigenvalues = ETA @ p
        q_full = 0.25 * (ETA @ (1.0 / eigenvalues))
        q_exp0 = exp_window_quasi_prob(p, beta=0.0)
        diff = np.abs(q_full - q_exp0).max()
        status = "OK" if diff < 1e-14 else "FAIL"
        table.add_row(name, f"{diff:.2e}", status)

    console.print(table)


if __name__ == "__main__":
    verify_beta_zero_is_full_pec()
