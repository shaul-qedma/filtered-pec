"""
PEC Filter Benchmark
====================

Compares PEC filter designs using pec_shared's realistic noise model.

NOTATION & METRICS
------------------
For quasi-probability q'(s), define:

  γ (gamma) := ||q'||_1 = Σ_s |q'(s)|
  
  This is the L1 norm of the quasi-probability distribution.
  For true probability distributions, γ = 1.
  For quasi-probabilities with negative values, γ > 1.

The PEC estimator is:
  
  Ô = γ × (1/N) Σᵢ sign(q'(sᵢ)) × O(sᵢ)

where sᵢ ~ |q'|/γ (sampling from normalized absolute values).

ERROR DECOMPOSITION
-------------------
  Bias     := E[Ô] - O_ideal           (systematic error)
  Variance := Var[Ô] = γ² × Var[raw] / N  (statistical error)
  RMSE     := √(Bias² + Variance)      (total error)

The key insight: Variance scales as γ²/N, so:
  
  Effective Samples := N / γ²

This is the equivalent number of unweighted samples.

FILTERS COMPARED
----------------
  Full PEC:     H(σ) = Λ(σ)⁻¹              (unbiased, high γ)
  Exp(β):       H(σ) = Λ(σ)⁻¹ e^{-β|σ|}   (biased, low γ)
  Tikhonov(α):  H(σ) = (Λ(σ) + α)⁻¹       (biased, medium γ)
  Natural(w):   H(σ) = Λ(σ)⁻¹ for |σ|≤w   (truncated, high γ)
  Modified(w):  H(σ) = Λ(σ)⁻¹ for |σ|≤w, else 1
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable
import time

from pec_shared import (
    ETA, STANDARD_GATES, NoisySimulator, Circuit,
    error_locations, random_circuit, random_noise_model,
    random_product_state, random_observable
)


# =============================================================================
# PRODUCT FILTERS
# =============================================================================
# These factorize as q'(s) = Π_v q'_v(s_v), enabling IID sampling.
# Each function takes noise probabilities p and returns local quasi-probs.

def filter_full(p: np.ndarray) -> np.ndarray:
    """
    Full PEC: H(σ) = Λ(σ)⁻¹
    
    Unbiased but high γ due to negative quasi-probabilities.
    
    Local filter: h_v(σ_v) = λ_v(σ_v)⁻¹
    Local quasi-prob: q_v(s) = (1/4) Σ_{σ_v} h_v(σ_v) η[s,σ_v]
    """
    lam = ETA @ p  # eigenvalues λ(0), λ(1), λ(2), λ(3)
    h = 1.0 / lam  # filter coefficients
    return 0.25 * (ETA @ h)


def filter_exp(beta: float) -> Callable:
    """
    Exponential window: H(σ) = Λ(σ)⁻¹ × e^{-β|σ|}
    
    Suppresses high-weight corrections exponentially.
    For β ≥ β_crit ≈ 1-λ, all quasi-probs become non-negative → γ = 1.
    
    Local filter: h_v(σ_v) = e^{-β·𝟙[σ_v≠0]} / λ_v(σ_v)
    """
    def f(p: np.ndarray) -> np.ndarray:
        lam = ETA @ p
        decay = np.exp(-beta)
        h = np.array([1.0/lam[0],          # σ=0: no decay
                      decay/lam[1],         # σ=1: decay
                      decay/lam[2],         # σ=2: decay
                      decay/lam[3]])        # σ=3: decay
        return 0.25 * (ETA @ h)
    return f


def filter_tikhonov(alpha: float) -> Callable:
    """
    Local Tikhonov regularization: H(σ) = Π_v (λ_v(σ_v) + α)⁻¹
    
    Adds regularization α to each eigenvalue before inversion.
    Reduces γ but introduces bias at all weights including σ=0.
    
    Local filter: h_v(σ_v) = (λ_v(σ_v) + α)⁻¹
    """
    def f(p: np.ndarray) -> np.ndarray:
        lam = ETA @ p
        h = 1.0 / (lam + alpha)
        return 0.25 * (ETA @ h)
    return f


# =============================================================================
# NON-PRODUCT FILTERS  
# =============================================================================
# These depend on total weight |σ| = Σ_v 𝟙[σ_v≠0], breaking factorization.
# Require enumeration over all 4^n configurations (expensive).

def filter_natural_truncation(w: int) -> Callable:
    """
    Natural truncation: H(σ) = Λ(σ)⁻¹ for |σ|≤w, else 0
    
    Completely drops high-weight Pauli paths.
    Bias = Σ_{|σ|>w} f̂(σ) where f̂ are path amplitudes.
    """
    def H(sigma: tuple, lambdas: list) -> float:
        weight = sum(1 for s in sigma if s != 0)
        if weight > w:
            return 0.0
        result = 1.0
        for v, s in enumerate(sigma):
            if s != 0:
                result /= lambdas[v][s]
        return result
    return H


def filter_modified_truncation(w: int) -> Callable:
    """
    Modified truncation: H(σ) = Λ(σ)⁻¹ for |σ|≤w, else 1
    
    Keeps noisy (uncompensated) high-weight paths instead of dropping.
    Bias = Σ_{|σ|>w} (1 - Λ(σ)) f̂(σ), smaller than natural truncation.
    """
    def H(sigma: tuple, lambdas: list) -> float:
        weight = sum(1 for s in sigma if s != 0)
        if weight <= w:
            result = 1.0
            for v, s in enumerate(sigma):
                if s != 0:
                    result /= lambdas[v][s]
            return result
        return 1.0
    return H


# =============================================================================
# PEC ESTIMATION
# =============================================================================

@dataclass
class PECResult:
    """
    Result of PEC estimation.
    
    Attributes:
        estimate: γ × mean(raw), the PEC estimate
        gamma: ||q'||_1, the L1 norm of quasi-probability
        raw_variance: Var[sign(q') × O], variance before γ scaling
        time: Wall-clock time in seconds
    """
    estimate: float
    gamma: float
    raw_variance: float
    time: float


def pec_product_filter(sim: NoisySimulator, circuit: Circuit, obs: str,
                       init: np.ndarray, locs: List[Tuple],
                       local_filter: Callable, n_samples: int, 
                       seed: int) -> PECResult:
    """
    PEC estimation with a product filter (IID sampling).
    
    Args:
        sim: Simulator instance
        circuit: Quantum circuit
        obs: Observable string (e.g., "XZI")
        init: Initial state vector
        locs: Error locations [(layer, qubit, noise_probs), ...]
        local_filter: Function p → q_v(s), local quasi-probability
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
    
    Returns:
        PECResult with estimate, gamma, raw_variance, time
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    
    # Compute local quasi-probabilities q_v(s) for each error location
    qp_list = [local_filter(p) for (_, _, p) in locs]
    
    # γ = ||q'||_1 = Π_v ||q_v||_1 (product of local L1 norms)
    local_norms = [np.abs(qp).sum() for qp in qp_list]
    gamma = np.prod(local_norms)
    
    # Sampling distribution: π_v(s) = |q_v(s)| / ||q_v||_1
    sampling_probs = [np.abs(qp) / norm for qp, norm in zip(qp_list, local_norms)]
    sampling_signs = [np.sign(qp) for qp in qp_list]
    
    # Monte Carlo sampling
    raw_estimates = np.empty(n_samples)
    for i in range(n_samples):
        insertions = {}
        sign = 1.0
        for v, (layer, qubit, _) in enumerate(locs):
            s = rng.choice(4, p=sampling_probs[v])
            insertions[(layer, qubit)] = s
            sign *= sampling_signs[v][s]
        
        raw_estimates[i] = sign * sim.run(circuit, obs, init, insertions)
    
    return PECResult(
        estimate=gamma * raw_estimates.mean(),
        gamma=float(gamma),
        raw_variance=raw_estimates.var(),
        time=time.time() - t0
    )


def pec_global_filter(sim: NoisySimulator, circuit: Circuit, obs: str,
                      init: np.ndarray, locs: List[Tuple],
                      filter_H: Callable, max_weight: int,
                      n_samples: int, seed: int) -> PECResult:
    """
    PEC estimation with a non-product filter (global enumeration + sampling).
    
    Enumerates all 4^n configurations to compute q'(s), then samples.
    Only feasible for small n (≤6 error locations).
    
    Args:
        filter_H: Function (sigma, lambdas) → H(sigma), the filter response
        max_weight: Maximum |σ| to include (for truncation filters)
    """
    from itertools import product as cartesian
    
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = len(locs)
    
    # Extract eigenvalues at each location
    lambdas = [ETA @ p for (_, _, p) in locs]
    
    # Enumerate q'(s) = 4^{-n} Σ_σ H(σ) χ_σ(s) for all s ∈ {0,1,2,3}^n
    configs = []  # List of (s, q'(s))
    for s in cartesian(range(4), repeat=n):
        val = 0.0
        for sigma in cartesian(range(4), repeat=n):
            weight = sum(1 for sv in sigma if sv != 0)
            if weight > max_weight:
                continue
            
            H = filter_H(sigma, lambdas)
            
            # χ_σ(s) = Π_v η[s_v, σ_v]
            chi = 1.0
            for sv, sigv in zip(s, sigma):
                chi *= ETA[sv, sigv]
            
            val += H * chi
        
        configs.append((s, val * (0.25 ** n)))
    
    # γ = ||q'||_1 = Σ_s |q'(s)|
    abs_vals = np.array([abs(v) for _, v in configs])
    gamma = abs_vals.sum()
    
    # Sampling distribution: π(s) = |q'(s)| / γ
    sampling_probs = abs_vals / gamma
    sampling_signs = np.array([1 if v >= 0 else -1 for _, v in configs])
    
    # Monte Carlo sampling
    raw_estimates = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.choice(len(configs), p=sampling_probs)
        s, _ = configs[idx]
        insertions = {(locs[v][0], locs[v][1]): s[v] for v in range(n)}
        raw_estimates[i] = sampling_signs[idx] * sim.run(circuit, obs, init, insertions)
    
    return PECResult(
        estimate=gamma * raw_estimates.mean(),
        gamma=gamma,
        raw_variance=raw_estimates.var(),
        time=time.time() - t0
    )


# =============================================================================
# BENCHMARK INFRASTRUCTURE
# =============================================================================

def run_trial(n_qubits: int, depth: int, n_samples: int, seed: int) -> dict:
    """
    Run single trial: generate random instance, run all filters, compare to ideal.
    
    Returns dict with results for each filter including bias, variance, gamma.
    """
    rng = np.random.default_rng(seed)
    
    # Generate random quantum instance
    circuit = random_circuit(n_qubits, depth, rng)
    noise = random_noise_model(rng)
    sim = NoisySimulator(STANDARD_GATES, noise)
    init = random_product_state(n_qubits, rng)
    obs = random_observable(n_qubits, rng)
    locs = error_locations(circuit, noise)
    n_locs = len(locs)
    
    # Compute ideal expectation (no noise)
    ideal = sim.ideal(circuit, obs, init)
    
    results = {'ideal': ideal, 'n_locs': n_locs, 'n_samples': n_samples}
    
    # -------------------------------------------------------------------------
    # Product filters (IID sampling, fast)
    # -------------------------------------------------------------------------
    product_filters = [
        ('Full PEC',     filter_full),
        ('Exp(β=0.1)',   filter_exp(0.1)),
        ('Exp(β=0.2)',   filter_exp(0.2)),
        ('Exp(β=0.3)',   filter_exp(0.3)),
        ('Tikh(α=0.05)', filter_tikhonov(0.05)),
        ('Tikh(α=0.1)',  filter_tikhonov(0.1)),
    ]
    
    for name, filt in product_filters:
        r = pec_product_filter(sim, circuit, obs, init, locs, filt, n_samples, seed)
        
        bias = r.estimate - ideal
        # Var[Ô] = γ² × Var[raw] / N
        variance = (r.gamma ** 2) * r.raw_variance / n_samples
        
        results[name] = {
            'bias': bias,
            'variance': variance,
            'gamma': r.gamma,
        }
    
    # -------------------------------------------------------------------------
    # Non-product filters (global enumeration, only for small n)
    # -------------------------------------------------------------------------
    if n_locs <= 6:
        global_filters = [
            ('Natural(w=1)',  filter_natural_truncation(1), 1),
            ('Natural(w=2)',  filter_natural_truncation(2), 2),
            ('Modified(w=1)', filter_modified_truncation(1), 1),
            ('Modified(w=2)', filter_modified_truncation(2), 2),
        ]
        
        for name, filt, w in global_filters:
            r = pec_global_filter(sim, circuit, obs, init, locs, filt, w, n_samples, seed)
            
            bias = r.estimate - ideal
            variance = (r.gamma ** 2) * r.raw_variance / n_samples
            
            results[name] = {
                'bias': bias,
                'variance': variance,
                'gamma': r.gamma,
            }
    
    return results


def run_benchmark(n_qubits: int, depth: int, n_samples: int, 
                  n_trials: int, seed: int = 42) -> dict:
    """
    Run benchmark: multiple trials, aggregate statistics.
    
    Returns summary dict with mean bias, variance, gamma, RMSE for each filter.
    """
    all_results = [run_trial(n_qubits, depth, n_samples, seed + t) 
                   for t in range(n_trials)]
    
    # Collect all filter names
    filter_names = set()
    for r in all_results:
        filter_names.update(k for k in r.keys() 
                           if k not in ['ideal', 'n_locs', 'n_samples'])
    
    summary = {
        'n_locs': all_results[0]['n_locs'],
        'n_samples': n_samples,
        'n_trials': n_trials,
    }
    
    for name in filter_names:
        data = [r[name] for r in all_results if name in r]
        if not data:
            continue
        
        biases = np.array([d['bias'] for d in data])
        variances = np.array([d['variance'] for d in data])
        gammas = np.array([d['gamma'] for d in data])
        
        mean_bias = np.mean(biases)
        mean_var = np.mean(variances)
        mean_gamma = np.mean(gammas)
        
        # RMSE = √(Bias² + Variance)
        rmse = np.sqrt(mean_bias**2 + mean_var)
        
        # Effective samples = N / γ²
        eff_samples = n_samples / np.mean(gammas**2)
        
        summary[name] = {
            'bias': mean_bias,
            'variance': mean_var,
            'gamma': mean_gamma,
            'rmse': rmse,
            'eff_samples': eff_samples,
        }
    
    return summary


def print_summary(config: tuple, summary: dict):
    """Print formatted results table."""
    nq, d, ns, nt = config
    
    print(f"\n{'='*85}")
    print(f"{nq} qubits, depth {d}, {summary['n_locs']} error locations")
    print(f"{ns} samples × {nt} trials")
    print(f"{'='*85}")
    print(f"{'Filter':<18} {'γ':>8} {'N_eff':>10} {'|Bias|':>10} {'√Var':>10} {'RMSE':>10}")
    print("-" * 70)
    
    # Extract filter results, sort by RMSE
    filters = [(k, v) for k, v in summary.items() 
               if k not in ['n_locs', 'n_samples', 'n_trials']]
    filters.sort(key=lambda x: x[1]['rmse'])
    
    for name, stats in filters:
        print(f"{name:<18} "
              f"{stats['gamma']:>8.2f} "
              f"{stats['eff_samples']:>10.1f} "
              f"{abs(stats['bias']):>10.5f} "
              f"{np.sqrt(stats['variance']):>10.5f} "
              f"{stats['rmse']:>10.5f}")


def main():
    """Run full benchmark suite."""
    
    print("=" * 85)
    print("PEC FILTER BENCHMARK")
    print("=" * 85)
    print(__doc__)
    
    # Circuit configurations to test
    configs = [
        # (n_qubits, depth, n_samples, n_trials)
        (2, 2, 500, 15),
        (3, 2, 500, 12),
        (4, 2, 500, 10),
        (4, 3, 400, 8),
        (5, 2, 400, 8),
    ]
    
    all_summaries = []
    for cfg in configs:
        print(f"\nRunning {cfg[0]}q depth={cfg[1]}...")
        summary = run_benchmark(*cfg)
        all_summaries.append((cfg, summary))
        print_summary(cfg, summary)
    
    # =========================================================================
    # AGGREGATE RESULTS
    # =========================================================================
    print("\n" + "=" * 85)
    print("AGGREGATE RESULTS ACROSS ALL CONFIGURATIONS")
    print("=" * 85)
    print("""
    γ (gamma)   = ||q'||_1, L1 norm of quasi-probability
    N_eff       = N / γ², effective sample count  
    |Bias|      = |E[Ô] - O_ideal|, systematic error
    √Var        = √Var[Ô], statistical error (std dev)
    RMSE        = √(Bias² + Var), total error
    """)
    
    # Collect all filter names
    filter_names = set()
    for _, s in all_summaries:
        filter_names.update(k for k in s.keys() 
                           if k not in ['n_locs', 'n_samples', 'n_trials'])
    
    # Compute averages across configurations
    print(f"{'Filter':<18} {'Avg γ':>8} {'Avg |Bias|':>12} {'Avg √Var':>12} {'Avg RMSE':>12}")
    print("-" * 65)
    
    method_stats = []
    for name in filter_names:
        data = [s[name] for _, s in all_summaries if name in s]
        if data:
            avg_gamma = np.mean([d['gamma'] for d in data])
            avg_bias = np.mean([abs(d['bias']) for d in data])
            avg_std = np.mean([np.sqrt(d['variance']) for d in data])
            avg_rmse = np.mean([d['rmse'] for d in data])
            method_stats.append((name, avg_gamma, avg_bias, avg_std, avg_rmse))
    
    # Sort by RMSE
    method_stats.sort(key=lambda x: x[4])
    
    for name, gamma, bias, std, rmse in method_stats:
        print(f"{name:<18} {gamma:>8.2f} {bias:>12.5f} {std:>12.5f} {rmse:>12.5f}")
    
    # =========================================================================
    # PARETO FRONTIER
    # =========================================================================
    print("\n" + "-" * 65)
    print("PARETO FRONTIER (non-dominated in RMSE vs γ):")
    print("-" * 65)
    
    pareto = []
    for m in method_stats:
        is_dominated = any(m2[4] < m[4] and m2[1] < m[1] for m2 in method_stats)
        if not is_dominated:
            pareto.append(m)
    
    for name, gamma, _, _, rmse in sorted(pareto, key=lambda x: x[1]):
        print(f"  {name:<18} γ = {gamma:.2f}, RMSE = {rmse:.5f}")
    
    # =========================================================================
    # COMPARISON TO FULL PEC
    # =========================================================================
    full_pec = next(m for m in method_stats if m[0] == 'Full PEC')
    full_gamma, full_rmse = full_pec[1], full_pec[4]
    
    print("\n" + "-" * 65)
    print(f"COMPARISON TO FULL PEC (γ = {full_gamma:.2f}, RMSE = {full_rmse:.5f}):")
    print("-" * 65)
    
    print("\nFilters with LOWER RMSE (better accuracy):")
    for name, gamma, _, _, rmse in method_stats:
        if rmse < full_rmse:
            rmse_change = (rmse / full_rmse - 1) * 100
            gamma_change = (gamma / full_gamma - 1) * 100
            print(f"  {name:<18} RMSE {rmse_change:+.1f}%, γ {gamma_change:+.1f}%")
    
    print("\nFilters with LOWER γ (lower variance per sample):")
    for name, gamma, _, _, rmse in method_stats:
        if gamma < full_gamma:
            rmse_change = (rmse / full_rmse - 1) * 100
            gamma_change = (gamma / full_gamma - 1) * 100
            print(f"  {name:<18} γ {gamma_change:+.1f}%, RMSE {rmse_change:+.1f}%")


if __name__ == "__main__":
    main()