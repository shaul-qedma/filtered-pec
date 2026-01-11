"""
PEC Filter Benchmark - Using pec_shared infrastructure

Properly compares all filters using pec_shared's realistic noise model.

METRICS:
  - Bias: E[estimator] - ideal (systematic error)
  - Variance: Var[estimator] (statistical error)  
  - RMSE: √(Bias² + Variance) (total error)
  - γ (gamma): ||q'||_1, the L1 norm of quasi-probability
  - Effective samples: N_actual / γ² (variance scales as γ²/N)
  - MSE per sample: γ² × Var[raw] + Bias² (cost-adjusted error)
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable
import time

from pec_shared import (
    ETA, PAULI, STANDARD_GATES,
    Gate, Circuit, NoisySimulator,
    error_locations, random_circuit, random_noise_model,
    random_product_state, random_observable
)


# =============================================================================
# FILTER DEFINITIONS
# =============================================================================

def filter_full(p: np.ndarray) -> np.ndarray:
    """Full PEC: H(σ) = λ(σ)^{-1}"""
    lam = ETA @ p
    return 0.25 * (ETA @ (1.0 / lam))


def filter_exp(beta: float):
    """Exponential window: H(σ) = λ(σ)^{-1} e^{-β|σ|}"""
    def f(p: np.ndarray) -> np.ndarray:
        lam = ETA @ p
        decay = np.exp(-beta)
        h = np.array([1.0/lam[0], decay/lam[1], decay/lam[2], decay/lam[3]])
        return 0.25 * (ETA @ h)
    return f


def filter_tikhonov(alpha: float):
    """Local Tikhonov: H(σ) = (λ(σ) + α)^{-1}"""
    def f(p: np.ndarray) -> np.ndarray:
        lam = ETA @ p
        h = 1.0 / (lam + alpha)
        return 0.25 * (ETA @ h)
    return f


# =============================================================================
# NON-PRODUCT FILTERS (weight-based truncation)
# =============================================================================

def compute_global_quasi_prob(locs: List[Tuple], filter_fn: Callable, 
                               max_weight: int = 0) -> Tuple[dict, float]:
    """
    Compute quasi-probability for non-product filters.
    
    Returns:
        configs: dict mapping s tuple -> q'(s) value
        overhead: ||q'||_1
    """
    n = len(locs)
    if max_weight is None:
        max_weight = n
    
    # Get eigenvalues at each location
    lambdas = [ETA @ p for (_, _, p) in locs]
    
    # Enumerate all s configurations
    from itertools import product as cartesian
    
    configs = {}
    for s in cartesian(range(4), repeat=n):
        # Compute q'(s) = 4^{-n} sum_sigma H(sigma) chi_sigma(s)
        val = 0.0
        for sigma in cartesian(range(4), repeat=n):
            weight = sum(1 for sv in sigma if sv != 0)
            if weight > max_weight:
                continue
            
            # H(sigma) from filter
            H = filter_fn(sigma, lambdas)
            
            # chi_sigma(s) = prod_v eta[s_v, sigma_v]
            chi = 1.0
            for sv, sigv in zip(s, sigma):
                chi *= ETA[sv, sigv]
            
            val += H * chi
        
        configs[s] = val * (0.25 ** n)
    
    overhead = sum(abs(v) for v in configs.values())
    return configs, overhead


def filter_natural_truncation(w: int):
    """Natural truncation: H(σ) = λ(σ)^{-1} for |σ|≤w, else 0"""
    def H(sigma: tuple, lambdas: list) -> float:
        weight = sum(1 for sv in sigma if sv != 0)
        if weight > w:
            return 0.0
        prod = 1.0
        for v, sv in enumerate(sigma):
            if sv != 0:
                prod *= 1.0 / lambdas[v][sv]
        return prod
    return H


def filter_modified_truncation(w: int):
    """Modified truncation: H(σ) = λ(σ)^{-1} for |σ|≤w, else 1"""
    def H(sigma: tuple, lambdas: list) -> float:
        weight = sum(1 for sv in sigma if sv != 0)
        if weight <= w:
            prod = 1.0
            for v, sv in enumerate(sigma):
                if sv != 0:
                    prod *= 1.0 / lambdas[v][sv]
            return prod
        else:
            return 1.0
    return H


# =============================================================================
# PEC ESTIMATOR
# =============================================================================

@dataclass
class PECResult:
    estimate: float
    gamma: float          # ||q'||_1, the L1 norm
    raw_variance: float   # Var of sign*O before scaling by gamma
    time: float


def pec_product_filter(sim: NoisySimulator, circuit: Circuit, obs: str,
                       init: np.ndarray, locs: List[Tuple],
                       filter_fn: Callable, n_samples: int, 
                       seed: int) -> PECResult:
    """PEC with product filter (IID sampling)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    
    # Compute local quasi-probabilities
    qp_list = [filter_fn(p) for (_, _, p) in locs]
    
    # γ = ||q'||_1 = ∏_v ||q_v||_1
    norms = [np.abs(qp).sum() for qp in qp_list]
    gamma = np.prod(norms)
    probs = [np.abs(qp) / n for qp, n in zip(qp_list, norms)]
    signs = [np.sign(qp) for qp in qp_list]
    
    # Sample raw estimates (before scaling by gamma)
    raw_estimates = []
    for _ in range(n_samples):
        insertions = {}
        sign = 1.0
        for v, (layer, qubit, _) in enumerate(locs):
            s = rng.choice(4, p=probs[v])
            insertions[(layer, qubit)] = s
            sign *= signs[v][s]
        raw_estimates.append(sign * sim.run(circuit, obs, init, insertions))
    
    raw_estimates = np.array(raw_estimates)
    
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
    """PEC with non-product filter (global sampling)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    n = len(locs)
    
    # Get eigenvalues
    lambdas = [ETA @ p for (_, _, p) in locs]
    
    # Compute all quasi-probabilities (expensive but exact)
    from itertools import product as cartesian
    
    configs = []
    for s in cartesian(range(4), repeat=n):
        val = 0.0
        for sigma in cartesian(range(4), repeat=n):
            weight = sum(1 for sv in sigma if sv != 0)
            if weight > max_weight:
                continue
            H = filter_H(sigma, lambdas)
            chi = 1.0
            for sv, sigv in zip(s, sigma):
                chi *= ETA[sv, sigv]
            val += H * chi
        configs.append((s, val * (0.25 ** n)))
    
    # γ = ||q'||_1
    probs = np.array([abs(v) for _, v in configs])
    gamma = probs.sum()
    probs = probs / gamma
    signs = np.array([1 if v >= 0 else -1 for _, v in configs])
    
    # Sample raw estimates
    raw_estimates = []
    for _ in range(n_samples):
        idx = rng.choice(len(configs), p=probs)
        s, _ = configs[idx]
        insertions = {(locs[v][0], locs[v][1]): s[v] for v in range(n)}
        raw_estimates.append(signs[idx] * sim.run(circuit, obs, init, insertions))
    
    raw_estimates = np.array(raw_estimates)
    
    return PECResult(
        estimate=gamma * raw_estimates.mean(),
        gamma=gamma,
        raw_variance=raw_estimates.var(),
        time=time.time() - t0
    )


# =============================================================================
# BENCHMARK
# =============================================================================

def run_trial(n_qubits: int, depth: int, n_samples: int, seed: int):
    """Run single trial comparing all filters."""
    rng = np.random.default_rng(seed)
    
    # Generate instance
    circuit = random_circuit(n_qubits, depth, rng)
    noise = random_noise_model(rng)
    sim = NoisySimulator(STANDARD_GATES, noise)
    init = random_product_state(n_qubits, rng)
    obs = random_observable(n_qubits, rng)
    locs = error_locations(circuit, noise)
    n_locs = len(locs)
    
    ideal = sim.ideal(circuit, obs, init)
    
    results = {'ideal': ideal, 'n_locs': n_locs, 'n_samples': n_samples}
    
    # Product filters
    product_filters = [
        ('Full PEC', filter_full),
        ('Exp(β=0.1)', filter_exp(0.1)),
        ('Exp(β=0.2)', filter_exp(0.2)),
        ('Exp(β=0.3)', filter_exp(0.3)),
        ('Tikh(α=0.05)', filter_tikhonov(0.05)),
        ('Tikh(α=0.1)', filter_tikhonov(0.1)),
    ]
    
    for name, filt in product_filters:
        r = pec_product_filter(sim, circuit, obs, init, locs, filt, n_samples, seed)
        bias = r.estimate - ideal
        # Variance of final estimate = γ² × Var[raw] / N
        variance = (r.gamma ** 2) * r.raw_variance / n_samples
        results[name] = {
            'bias': bias,
            'variance': variance,
            'gamma': r.gamma,
            'raw_var': r.raw_variance
        }
    
    # Non-product filters (only if small enough)
    if n_locs <= 6:
        global_filters = [
            ('Natural(w=1)', filter_natural_truncation(1), 1),
            ('Natural(w=2)', filter_natural_truncation(2), 2),
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
                'raw_var': r.raw_variance
            }
    
    return results


def run_benchmark(n_qubits: int, depth: int, n_samples: int, n_trials: int, seed: int = 42):
    """Run full benchmark."""
    all_results = []
    for t in range(n_trials):
        all_results.append(run_trial(n_qubits, depth, n_samples, seed + t))
    
    # Aggregate
    methods = set()
    for r in all_results:
        methods.update(k for k in r.keys() if k not in ['ideal', 'n_locs', 'n_samples'])
    
    summary = {'n_locs': all_results[0]['n_locs'], 'n_samples': n_samples}
    for m in methods:
        data = [r[m] for r in all_results if m in r]
        if data:
            biases = np.array([d['bias'] for d in data])
            variances = np.array([d['variance'] for d in data])
            gammas = np.array([d['gamma'] for d in data])
            
            mean_bias = np.mean(biases)
            mean_var = np.mean(variances)
            rmse = np.sqrt(mean_bias**2 + mean_var)  # Bias² + Variance
            
            summary[m] = {
                'bias': mean_bias,
                'bias_std': np.std(biases),
                'variance': mean_var,
                'rmse': rmse,
                'gamma': np.mean(gammas),
                'eff_samples': n_samples / np.mean(gammas**2)  # N / γ²
            }
    
    return summary


def print_summary(config: tuple, summary: dict):
    """Print formatted results."""
    nq, d, ns, nt = config
    print(f"\n{'='*85}")
    print(f"{nq} qubits, depth {d}, {summary['n_locs']} locations | {ns} samples × {nt} trials")
    print(f"{'='*85}")
    print(f"{'Filter':<18} {'γ':>8} {'Eff.Samp':>10} {'|Bias|':>10} {'√Var':>10} {'RMSE':>10}")
    print("-" * 70)
    
    # Sort by RMSE
    methods = [(k, v) for k, v in summary.items() if k not in ['n_locs', 'n_samples']]
    methods.sort(key=lambda x: x[1]['rmse'])
    
    for name, stats in methods:
        sqrt_var = np.sqrt(stats['variance'])
        print(f"{name:<18} {stats['gamma']:>8.2f} {stats['eff_samples']:>10.1f} "
              f"{abs(stats['bias']):>10.5f} {sqrt_var:>10.5f} {stats['rmse']:>10.5f}")


def main():
    print("=" * 75)
    print("PEC FILTER BENCHMARK (using pec_shared)")
    print("=" * 75)
    
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
    
    # Aggregate across all configs
    print("\n" + "=" * 85)
    print("AGGREGATE RESULTS")
    print("=" * 85)
    print("""
γ (gamma) = ||q'||_1 : L1 norm of quasi-probability distribution
Eff.Samples = N / γ² : effective sample count (variance scales as γ²/N)
|Bias| : absolute systematic error
√Var : standard deviation of estimator  
RMSE = √(Bias² + Var) : total error
""")
    
    methods = set()
    for _, s in all_summaries:
        methods.update(k for k in s.keys() if k not in ['n_locs', 'n_samples'])
    
    print(f"{'Filter':<18} {'Avg γ':>8} {'Avg |Bias|':>12} {'Avg √Var':>12} {'Avg RMSE':>12}")
    print("-" * 65)
    
    method_stats = []
    for m in methods:
        data = [s[m] for _, s in all_summaries if m in s]
        if data:
            avg_gamma = np.mean([d['gamma'] for d in data])
            avg_bias = np.mean([abs(d['bias']) for d in data])
            avg_std = np.mean([np.sqrt(d['variance']) for d in data])
            avg_rmse = np.mean([d['rmse'] for d in data])
            method_stats.append((m, avg_gamma, avg_bias, avg_std, avg_rmse, len(data)))
    
    method_stats.sort(key=lambda x: x[4])  # sort by RMSE
    for name, gamma, bias, std, rmse, count in method_stats:
        print(f"{name:<18} {gamma:>8.2f} {bias:>12.5f} {std:>12.5f} {rmse:>12.5f}")
    
    # Find Pareto frontier (RMSE vs γ tradeoff)
    print("\n" + "-" * 65)
    print("PARETO FRONTIER (best RMSE vs γ tradeoffs):")
    pareto = []
    for m in method_stats:
        dominated = False
        for m2 in method_stats:
            if m2[4] < m[4] and m2[1] < m[1]:  # better RMSE AND lower gamma
                dominated = True
                break
        if not dominated:
            pareto.append(m)
    
    for name, gamma, bias, std, rmse, _ in sorted(pareto, key=lambda x: x[1]):
        print(f"  {name:<18} γ={gamma:.2f}, RMSE={rmse:.5f}")
    
    print("\n" + "=" * 85)
    print("RECOMMENDATIONS")
    print("=" * 85)
    
    # Find best by category
    best_rmse = min(method_stats, key=lambda x: x[4])
    best_gamma = min(method_stats, key=lambda x: x[1])
    
    print(f"\nLowest RMSE: {best_rmse[0]} (RMSE={best_rmse[4]:.5f}, γ={best_rmse[1]:.2f})")
    print(f"Lowest γ: {best_gamma[0]} (γ={best_gamma[1]:.2f}, RMSE={best_gamma[4]:.5f})")
    
    # Check if any filter beats Full PEC
    full_idx = next(i for i, m in enumerate(method_stats) if m[0] == 'Full PEC')
    full_gamma, full_rmse = method_stats[full_idx][1], method_stats[full_idx][4]
    
    print(f"\nFilters with lower RMSE than Full PEC ({full_rmse:.5f}):")
    for name, gamma, bias, std, rmse, _ in method_stats:
        if rmse < full_rmse:
            print(f"  {name}: RMSE={rmse:.5f} ({(1-rmse/full_rmse)*100:+.1f}%), γ={gamma:.2f}")
    
    print(f"\nFilters with lower γ than Full PEC ({full_gamma:.2f}):")
    for name, gamma, bias, std, rmse, _ in method_stats:
        if gamma < full_gamma:
            print(f"  {name}: γ={gamma:.2f} ({(1-gamma/full_gamma)*100:+.1f}%), RMSE={rmse:.5f}")


if __name__ == "__main__":
    main()