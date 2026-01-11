"""
Exponential Window PEC Sampler
==============================

PEC with exponential suppression of high-weight Pauli corrections.

The filter in Fourier domain:
    h[σ] = e^{-β·𝟙[σ≠0]} / (η·p)[σ]

For β = 0, this reduces to full PEC: h[σ] = 1/(η·p)[σ].
For β > 0, non-identity corrections are suppressed by e^{-β}.

The quasi-probability:
    q[s] = (1/4) (η · h)[s]
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from pec_shared import (
    ETA, STANDARD_GATES, NoisySimulator, Circuit,
    error_locations, random_circuit, random_noise_model,
    random_product_state, random_observable
)


# =============================================================================
# EXPONENTIAL WINDOW FILTER
# =============================================================================

def exp_window_quasi_prob(p: np.ndarray, beta: float) -> np.ndarray:
    """
    Compute local quasi-probability with exponential window.
    
    Args:
        p: Noise probabilities [p_I, p_X, p_Y, p_Z]
        beta: Suppression parameter (β = 0 gives full PEC)
    
    Returns:
        q: Quasi-probability [q_I, q_X, q_Y, q_Z]
    
    The filter h[σ] = e^{-β·𝟙[σ≠0]} / (η·p)[σ] gives:
        q = (1/4) η · h
    """
    eigenvalues = ETA @ p  # (η·p)[σ] for σ ∈ {0,1,2,3}
    
    decay = np.exp(-beta)
    h = np.array([1.0 / eigenvalues[0],           # σ=0: no suppression
                  decay / eigenvalues[1],          # σ=1: suppressed
                  decay / eigenvalues[2],          # σ=2: suppressed
                  decay / eigenvalues[3]])         # σ=3: suppressed
    
    return 0.25 * (ETA @ h)


def gamma(q: np.ndarray) -> float:
    """Sampling overhead γ = ||q||₁"""
    return np.abs(q).sum()


# =============================================================================
# PEC ESTIMATOR
# =============================================================================

@dataclass
class PECEstimate:
    """
    Result of PEC estimation.
    
    Attributes:
        mean: Estimated expectation value
        std: Standard error of the mean
        gamma: Total sampling overhead Π_v γ_v
        n_samples: Number of samples used
    """
    mean: float
    std: float
    gamma: float
    n_samples: int


def pec_estimate(
    sim: NoisySimulator,
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    error_locs: List[Tuple],
    beta: float,
    n_samples: int,
    seed: int = 0
) -> PECEstimate:
    """
    PEC estimation with exponential window filter.
    
    Args:
        sim: Noisy simulator instance
        circuit: Quantum circuit
        observable: Pauli string (e.g., "XZI")
        initial_state: Initial state vector
        error_locs: List of (layer, qubit, noise_probs) tuples
        beta: Exponential suppression parameter (0 = full PEC)
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
    
    Returns:
        PECEstimate with mean, std, gamma, n_samples
    """
    rng = np.random.default_rng(seed)
    
    # Compute local quasi-probabilities at each error location
    local_q = [exp_window_quasi_prob(p, beta) for (_, _, p) in error_locs]
    
    # Total γ = Π_v ||q_v||₁
    local_gamma = [gamma(q) for q in local_q]
    total_gamma = np.prod(local_gamma)
    
    # Sampling distributions: π_v[s] = |q_v[s]| / γ_v
    sampling_probs = [np.abs(q) / g for q, g in zip(local_q, local_gamma)]
    sampling_signs = [np.sign(q) for q in local_q]
    
    # Monte Carlo sampling
    estimates = np.empty(n_samples)
    
    for i in range(n_samples):
        # Sample Pauli insertions independently at each location
        insertions = {}
        sign = 1.0
        
        for v, (layer, qubit, _) in enumerate(error_locs):
            s = rng.choice(4, p=sampling_probs[v])
            insertions[(layer, qubit)] = s
            sign *= sampling_signs[v][s]
        
        # Run noisy simulation with insertions, weight by sign
        estimates[i] = sign * sim.run(circuit, observable, initial_state, insertions)
    
    # PEC estimate: γ × mean(raw estimates)
    mean = total_gamma * estimates.mean()
    std = total_gamma * estimates.std() / np.sqrt(n_samples)
    
    return PECEstimate(mean=mean, std=std, gamma=float(total_gamma), n_samples=n_samples)


# =============================================================================
# VERIFICATION
# =============================================================================

def verify_beta_zero_is_full_pec():
    """Verify that β=0 recovers full PEC quasi-probabilities."""
    print("Verification: β=0 equals Full PEC")
    print("=" * 50)
    
    # Test with various noise configurations
    test_cases = [
        ("Depolarizing 5%", np.array([0.95, 0.05/3, 0.05/3, 0.05/3])),
        ("Depolarizing 10%", np.array([0.90, 0.10/3, 0.10/3, 0.10/3])),
        ("Asymmetric", np.array([0.90, 0.05, 0.03, 0.02])),
        ("Z-dominated", np.array([0.85, 0.02, 0.03, 0.10])),
    ]
    
    for name, p in test_cases:
        # Full PEC: q = (1/4) η · (1/(η·p))
        eigenvalues = ETA @ p
        q_full = 0.25 * (ETA @ (1.0 / eigenvalues))
        
        # Exponential window with β=0
        q_exp0 = exp_window_quasi_prob(p, beta=0.0)
        
        # Check equality
        diff = np.abs(q_full - q_exp0).max()
        status = "✓" if diff < 1e-14 else "✗"
        print(f"{status} {name}: max|diff| = {diff:.2e}")
    
    print()


def demo_beta_effect():
    """Demonstrate effect of β on quasi-probabilities."""
    print("Effect of β on quasi-probabilities")
    print("=" * 50)
    
    p = np.array([0.90, 0.05, 0.03, 0.02])  # Asymmetric noise
    print(f"Noise: p = {p}")
    print(f"Eigenvalues: η·p = {(ETA @ p).round(4)}")
    print()
    
    print(f"{'β':<6} {'q[I]':>8} {'q[X]':>8} {'q[Y]':>8} {'q[Z]':>8} {'γ':>8} {'all≥0':>8}")
    print("-" * 60)
    
    for beta in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        q = exp_window_quasi_prob(p, beta)
        g = gamma(q)
        nonneg = "yes" if np.all(q >= -1e-10) else "no"
        print(f"{beta:<6.2f} {q[0]:>8.4f} {q[1]:>8.4f} {q[2]:>8.4f} {q[3]:>8.4f} {g:>8.4f} {nonneg:>8}")
    
    print()


def benchmark():
    """Compare Full PEC vs Exponential Window on random circuits."""
    print("Benchmark: Full PEC vs Exponential Window")
    print("=" * 50)
    
    rng = np.random.default_rng(42)
    n_samples = 500
    n_trials = 10
    
    configs = [
        (3, 2),  # 3 qubits, depth 2
        (4, 2),  # 4 qubits, depth 2
        (4, 3),  # 4 qubits, depth 3
    ]
    
    for n_qubits, depth in configs:
        print(f"\n{n_qubits} qubits, depth {depth}")
        print("-" * 40)
        
        results = {beta: [] for beta in [0.0, 0.1, 0.2]}
        
        for trial in range(n_trials):
            # Generate random instance
            circuit = random_circuit(n_qubits, depth, rng)
            noise = random_noise_model(rng)
            sim = NoisySimulator(STANDARD_GATES, noise)
            init = random_product_state(n_qubits, rng)
            obs = random_observable(n_qubits, rng)
            locs = error_locations(circuit, noise)
            
            ideal = sim.ideal(circuit, obs, init)
            
            for beta in results.keys():
                est = pec_estimate(sim, circuit, obs, init, locs, beta, n_samples, seed=trial)
                error = est.mean - ideal
                results[beta].append({
                    'error': error,
                    'gamma': est.gamma,
                })
        
        # Print summary
        print(f"{'β':<6} {'γ':>10} {'|Bias|':>12} {'RMSE':>12}")
        for beta, data in results.items():
            errors = np.array([d['error'] for d in data])
            gammas = np.array([d['gamma'] for d in data])
            bias = np.abs(errors.mean())
            rmse = np.sqrt((errors**2).mean())
            print(f"{beta:<6.1f} {gammas.mean():>10.2f} {bias:>12.5f} {rmse:>12.5f}")


if __name__ == "__main__":
    verify_beta_zero_is_full_pec()
    demo_beta_effect()
    benchmark()