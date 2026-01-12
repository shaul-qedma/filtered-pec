"""
Threshold Filter PEC via Importance Sampling from Exponential Window
=====================================================================

We sample insertions from an exponential-window proposal (product form)
and reweight by the ratio of global filters:

    h_prop(w) = exp(-beta_prop * w)
    h_target(w) = threshold/softplus filter on the global weight w

Estimator:
    E = qp_norm_proposal * mean[ sign(sigma) * O_sigma * h_target(w) / h_prop(w) ]

This keeps product sampling but applies a non-product target filter.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from exp_window_pec import (
    exp_window_quasi_prob,
    qp_norm,
    pec_estimate,
    pec_estimate_qiskit,
    _build_qiskit_circuit,
    _qiskit_noise_instructions,
    _require_qiskit,
    _to_qiskit_statevector,
)
from pec_shared import (
    ETA,
    STANDARD_GATES,
    Circuit,
    NoisySimulator,
    error_locations,
    random_circuit,
    random_noise_model,
    random_product_state,
    random_observable,
)


def h_exponential(weight: int, beta: float) -> float:
    return float(np.exp(-beta * weight))


def h_threshold(weight: int, beta: float, w0: int) -> float:
    if weight <= w0:
        return 1.0
    return float(np.exp(-beta * (weight - w0)))


def h_softplus(weight: int, beta: float, w0: float, tau: float = 1.0) -> float:
    x = (weight - w0) / tau
    x = float(np.clip(x, -20.0, 20.0))
    sp = np.log1p(np.exp(x)) if x <= 20.0 else x
    return float(np.exp(-beta * sp * tau))


def critical_beta(p: np.ndarray) -> float:
    """Critical beta where local q becomes non-negative."""
    eigenvalues = ETA @ p
    min_eigenvalue = min(eigenvalues[1], eigenvalues[2], eigenvalues[3])
    if min_eigenvalue <= 0:
        return float("inf")
    return float(-np.log(min_eigenvalue))


def _proposal_nonid_probs(error_locs: List[Tuple], beta_prop: float) -> np.ndarray:
    if not error_locs:
        return np.array([], dtype=float)
    probs = []
    for _, _, p in error_locs:
        q = exp_window_quasi_prob(p, beta_prop)
        g = qp_norm(q)
        prob = np.abs(q) / g
        probs.append(1.0 - float(prob[0]))
    return np.array(probs, dtype=float)


def _weight_pmf(p_nonid: np.ndarray) -> np.ndarray:
    pmf = np.array([1.0], dtype=float)
    for p in p_nonid:
        nxt = np.zeros(pmf.size + 1, dtype=float)
        nxt[:-1] += pmf * (1.0 - p)
        nxt[1:] += pmf * p
        pmf = nxt
    return pmf


def _expected_weight(p_nonid: np.ndarray) -> float:
    return float(np.sum(p_nonid)) if p_nonid.size else 0.0


def _resolve_w0(
    w0: int,
    w0_density: Optional[float],
    n_locs: int,
    w0_mode: str,
    expected_weight: Optional[float],
    p_nonid: Optional[np.ndarray],
) -> int:
    if w0_density is None:
        return int(max(0, w0))
    if w0_density < 0:
        raise ValueError("w0_density must be non-negative")
    if w0_mode == "expected":
        scale = expected_weight if expected_weight is not None else float(n_locs)
    elif w0_mode == "volume":
        scale = float(n_locs)
    elif w0_mode == "tail":
        if w0_density > 1:
            raise ValueError("w0_density must be <= 1 for w0_mode='tail'")
        if p_nonid is None or p_nonid.size == 0:
            return int(max(0, w0))
        pmf = _weight_pmf(p_nonid)
        cdf = np.cumsum(pmf)
        target = max(0.0, 1.0 - w0_density)
        scaled = int(np.searchsorted(cdf, target, side="left"))
        return int(min(max(0, scaled), n_locs))
    else:
        raise ValueError(f"Unknown w0_mode: {w0_mode}")
    scaled = int(np.ceil(w0_density * scale))
    return int(min(max(0, scaled), n_locs))


@dataclass
class ThresholdPECEstimate:
    mean: float
    std: float
    qp_norm_proposal: float
    effective_samples: float
    n_samples: int
    weight_mean: float
    weight_std: float
    weight_above_frac: float
    exp_mean: Optional[float] = None
    exp_std: Optional[float] = None


def _resolve_filter(
    filter_type: str,
    w0: int,
    beta_thresh: float,
    softplus_tau: float,
    h_target: Optional[Callable[[int], float]],
) -> Callable[[int], float]:
    if h_target is not None:
        return h_target
    if filter_type == "threshold":
        return lambda w: h_threshold(w, beta_thresh, w0)
    if filter_type == "softplus":
        return lambda w: h_softplus(w, beta_thresh, w0, softplus_tau)
    raise ValueError(f"Unknown filter type: {filter_type}")


def _standard_error(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    return float(values.std(ddof=1)) / np.sqrt(values.size)


def _importance_sampling_estimate(
    measure_func: Callable[[Dict[Tuple[int, int], int]], float],
    error_locs: List[Tuple],
    beta_prop: float,
    h_target: Callable[[int], float],
    n_samples: int,
    rng: np.random.Generator,
    w0_value: Optional[int],
) -> ThresholdPECEstimate:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    if not error_locs:
        base_val = float(measure_func({}))
        h0 = h_target(0)
        return ThresholdPECEstimate(
            mean=base_val * h0,
            std=0.0,
            qp_norm_proposal=1.0,
            effective_samples=float(n_samples),
            n_samples=n_samples,
            weight_mean=0.0,
            weight_std=0.0,
            weight_above_frac=0.0,
            exp_mean=base_val,
            exp_std=0.0,
        )

    local_q = [exp_window_quasi_prob(p, beta_prop) for (_, _, p) in error_locs]
    local_qp_norm = [qp_norm(q) for q in local_q]
    total_qp_norm = float(np.prod(local_qp_norm)) if local_qp_norm else 1.0

    sampling_probs = [np.abs(q) / g for q, g in zip(local_q, local_qp_norm)]
    sampling_signs = [np.sign(q) for q in local_q]

    exp_estimates = np.empty(n_samples, dtype=float)
    target_estimates = np.empty(n_samples, dtype=float)
    importance_weights = np.empty(n_samples, dtype=float)
    sample_weights = np.empty(n_samples, dtype=float)

    for i in range(n_samples):
        insertions: Dict[Tuple[int, int], int] = {}
        sign = 1.0
        weight = 0
        for v, (layer, qubit, _) in enumerate(error_locs):
            s = int(rng.choice(4, p=sampling_probs[v]))
            sign *= sampling_signs[v][s]
            if s != 0:
                insertions[(layer, qubit)] = s
                weight += 1

        measurement = float(measure_func(insertions))
        exp_val = sign * measurement

        h_prop = h_exponential(weight, beta_prop)
        h_targ = h_target(weight)
        iw = h_targ / h_prop if h_prop > 1e-15 else 0.0

        exp_estimates[i] = exp_val
        target_estimates[i] = iw * exp_val
        importance_weights[i] = iw
        sample_weights[i] = weight

    exp_mean = total_qp_norm * float(exp_estimates.mean())
    exp_std = total_qp_norm * _standard_error(exp_estimates)
    target_mean = total_qp_norm * float(target_estimates.mean())
    target_std = total_qp_norm * _standard_error(target_estimates)

    iw_mean = float(np.mean(importance_weights))
    iw_var = float(np.var(importance_weights))
    ess = n_samples / (1.0 + iw_var / (iw_mean ** 2)) if iw_mean > 1e-12 else 0.0

    above_frac = float(np.mean(sample_weights > w0_value)) if w0_value is not None else float("nan")

    return ThresholdPECEstimate(
        mean=target_mean,
        std=target_std,
        qp_norm_proposal=total_qp_norm,
        effective_samples=float(ess),
        n_samples=n_samples,
        weight_mean=float(sample_weights.mean()),
        weight_std=float(sample_weights.std(ddof=1)) if n_samples > 1 else 0.0,
        weight_above_frac=above_frac,
        exp_mean=exp_mean,
        exp_std=exp_std,
    )


def threshold_pec_estimate(
    sim: NoisySimulator,
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    error_locs: List[Tuple],
    w0: int,
    beta_prop: float,
    beta_thresh: float,
    n_samples: int,
    seed: int = 0,
    filter_type: str = "threshold",
    softplus_tau: float = 1.0,
    h_target: Optional[Callable[[int], float]] = None,
    w0_density: Optional[float] = None,
    w0_mode: str = "expected",
) -> ThresholdPECEstimate:
    rng = np.random.default_rng(seed)
    p_nonid = _proposal_nonid_probs(error_locs, beta_prop)
    expected_weight = _expected_weight(p_nonid)
    w0_value = _resolve_w0(
        w0, w0_density, len(error_locs), w0_mode, expected_weight, p_nonid
    )
    h_fn = _resolve_filter(filter_type, w0_value, beta_thresh, softplus_tau, h_target)

    def measure(insertions: Dict[Tuple[int, int], int]) -> float:
        return float(sim.run(circuit, observable, initial_state, insertions))

    return _importance_sampling_estimate(
        measure, error_locs, beta_prop, h_fn, n_samples, rng, w0_value
    )


def threshold_pec_qiskit(
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    noise: dict,
    error_locs: List[Tuple],
    w0: int,
    beta_prop: float,
    beta_thresh: float,
    n_samples: int,
    seed: int = 0,
    filter_type: str = "threshold",
    softplus_tau: float = 1.0,
    h_target: Optional[Callable[[int], float]] = None,
    w0_density: Optional[float] = None,
    w0_mode: str = "expected",
) -> ThresholdPECEstimate:
    rng = np.random.default_rng(seed)
    p_nonid = _proposal_nonid_probs(error_locs, beta_prop)
    expected_weight = _expected_weight(p_nonid)
    w0_value = _resolve_w0(
        w0, w0_density, len(error_locs), w0_mode, expected_weight, p_nonid
    )
    h_fn = _resolve_filter(filter_type, w0_value, beta_thresh, softplus_tau, h_target)

    QuantumCircuit, UnitaryGate, Pauli, DensityMatrix, Kraus, AerSimulator = _require_qiskit()
    init_sv = _to_qiskit_statevector(initial_state, circuit.n_qubits)
    init_density = DensityMatrix(init_sv)
    pauli_obs = Pauli(observable[::-1])
    noise_instr = _qiskit_noise_instructions(noise, Kraus)
    backend = AerSimulator(method="density_matrix")

    def measure(insertions: Dict[Tuple[int, int], int]) -> float:
        qc = _build_qiskit_circuit(
            circuit=circuit,
            init_density=init_density,
            noise_instr=noise_instr,
            insertions=insertions,
            QuantumCircuit=QuantumCircuit,
            UnitaryGate=UnitaryGate,
        )
        qc.save_density_matrix()
        result = backend.run(qc).result()
        rho = result.data(0)["density_matrix"]
        dm = DensityMatrix(rho)
        return float(np.real(dm.expectation_value(pauli_obs)))

    return _importance_sampling_estimate(
        measure, error_locs, beta_prop, h_fn, n_samples, rng, w0_value
    )


def benchmark(
    n_qubits: int,
    depth: int,
    w0: int,
    beta_prop: float,
    beta_thresh: float,
    n_samples: int,
    n_trials: int,
    seed: int,
    filter_type: str = "threshold",
    softplus_tau: float = 1.0,
    use_qiskit: bool = False,
    w0_density: Optional[float] = None,
    w0_mode: str = "volume",
) -> Dict:
    results = {
        "full_pec": [],
        "exp_window": [],
        "filtered": [],
    }
    ideals = []

    w0_values: List[int] = []
    expected_weights: List[float] = []
    tail_targets: List[float] = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        circuit = random_circuit(n_qubits, depth, rng)
        noise = random_noise_model(rng)
        init = random_product_state(n_qubits, rng)
        obs = random_observable(n_qubits, rng)
        locs = error_locations(circuit, noise)
        p_nonid = _proposal_nonid_probs(locs, beta_prop)
        expected_weight = _expected_weight(p_nonid)
        expected_weights.append(expected_weight)
        w0_value = _resolve_w0(
            w0, w0_density, len(locs), w0_mode, expected_weight, p_nonid
        )
        w0_values.append(w0_value)
        tail_targets.append(float(w0_density) if w0_mode == "tail" and w0_density is not None else float("nan"))

        sim = NoisySimulator(STANDARD_GATES, noise)
        ideal = sim.ideal(circuit, obs, init)
        ideals.append(ideal)

        if use_qiskit:
            est_full = pec_estimate_qiskit(
                circuit, obs, init, noise, locs, beta=0.0, n_samples=n_samples, seed=seed + 1000 * t
            )
            est_exp = pec_estimate_qiskit(
                circuit, obs, init, noise, locs, beta=beta_prop, n_samples=n_samples, seed=seed + 2000 * t
            )
            est_filtered = threshold_pec_qiskit(
                circuit,
                obs,
                init,
                noise,
                locs,
                w0=w0_value,
                beta_prop=beta_prop,
                beta_thresh=beta_thresh,
                n_samples=n_samples,
                seed=seed + 3000 * t,
                filter_type=filter_type,
                softplus_tau=softplus_tau,
                w0_mode=w0_mode,
            )
        else:
            est_full = pec_estimate(
                sim, circuit, obs, init, locs, beta=0.0, n_samples=n_samples, seed=seed + 1000 * t
            )
            est_exp = pec_estimate(
                sim, circuit, obs, init, locs, beta=beta_prop, n_samples=n_samples, seed=seed + 2000 * t
            )
            est_filtered = threshold_pec_estimate(
                sim,
                circuit,
                obs,
                init,
                locs,
                w0=w0_value,
                beta_prop=beta_prop,
                beta_thresh=beta_thresh,
                n_samples=n_samples,
                seed=seed + 3000 * t,
                filter_type=filter_type,
                softplus_tau=softplus_tau,
                w0_mode=w0_mode,
            )

        results["full_pec"].append(
            {"estimate": est_full.mean, "qp_norm": est_full.qp_norm, "error": est_full.mean - ideal}
        )
        results["exp_window"].append(
            {"estimate": est_exp.mean, "qp_norm": est_exp.qp_norm, "error": est_exp.mean - ideal}
        )
        results["filtered"].append(
            {
                "estimate": est_filtered.mean,
                "qp_norm": est_filtered.qp_norm_proposal,
                "ess": est_filtered.effective_samples,
                "above": est_filtered.weight_above_frac,
                "weight_mean": est_filtered.weight_mean,
                "weight_std": est_filtered.weight_std,
                "error": est_filtered.mean - ideal,
            }
        )

    return {
        "results": results,
        "ideals": ideals,
        "config": {
            "n_qubits": n_qubits,
            "depth": depth,
            "w0": w0,
            "w0_density": w0_density,
            "w0_mode": w0_mode,
            "w0_mean": float(np.mean(w0_values)) if w0_values else 0.0,
            "w0_min": int(min(w0_values)) if w0_values else 0,
            "w0_max": int(max(w0_values)) if w0_values else 0,
            "expected_weight_mean": float(np.mean(expected_weights)) if expected_weights else 0.0,
            "tail_target": float(np.mean(tail_targets)) if tail_targets else float("nan"),
            "beta_prop": beta_prop,
            "beta_thresh": beta_thresh,
            "n_samples": n_samples,
            "n_trials": n_trials,
            "filter_type": filter_type,
            "softplus_tau": softplus_tau,
            "use_qiskit": use_qiskit,
        },
    }


def print_benchmark_results(data: Dict) -> None:
    results = data["results"]
    config = data["config"]
    label = "threshold"
    if config["filter_type"] == "softplus":
        label = f"softplus τ={config['softplus_tau']}"

    w0_label = f"w0={config['w0']}"
    if config.get("w0_density") is not None:
        exp_label = ""
        if config.get("w0_mode") == "expected":
            exp_label = f", E[w]≈{config.get('expected_weight_mean', 0.0):.1f}"
        if config.get("w0_mode") == "tail":
            exp_label = f", target p_gt_w0≈{config.get('tail_target', 0.0):.2f}"
        w0_label = (
            f"w0≈{config['w0_mean']:.1f} "
            f"(ρ={config['w0_density']}, range {config['w0_min']}-{config['w0_max']}{exp_label})"
        )
        w0_label = f"{w0_label}, mode={config.get('w0_mode', 'expected')}"

    print(
        f"\nConfig: {config['n_qubits']}q, depth={config['depth']}, {w0_label}, "
        f"β_prop={config['beta_prop']}, β_thresh={config['beta_thresh']}, filter={label}"
    )
    print(f"Samples: {config['n_samples']}, Trials: {config['n_trials']}, Qiskit={config['use_qiskit']}")

    print(f"\n{'Method':<20} {'qp_norm':>8} {'Bias':>10} {'RMSE':>10} {'ESS':>10} {'w_mean':>8} {'p_gt_w0':>8}")
    print("-" * 78)

    for method in ["full_pec", "exp_window", "filtered"]:
        data_m = results[method]
        errors = np.array([d["error"] for d in data_m])
        qp_norms = np.array([d["qp_norm"] for d in data_m])

        bias = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_qp_norm = float(np.mean(qp_norms))

        if method == "filtered":
            ess = float(np.mean([d["ess"] for d in data_m]))
            name = label
            w_mean = float(np.mean([d.get("weight_mean") for d in data_m if "weight_mean" in d]))
            above = float(np.mean([d.get("above") for d in data_m if "above" in d]))
            print(
                f"{name:<20} {mean_qp_norm:>8.2f} {bias:>10.4f} {rmse:>10.4f} {ess:>10.1f} "
                f"{w_mean:>8.2f} {above:>8.2f}"
            )
        else:
            name = "full_pec" if method == "full_pec" else "exp_window"
            print(
                f"{name:<20} {mean_qp_norm:>8.2f} {bias:>10.4f} {rmse:>10.4f} "
                f"{'N/A':>10} {'-':>8} {'-':>8}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold Filter PEC via Importance Sampling")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--w0", type=int, default=2)
    parser.add_argument("--w0-density", type=float, default=None,
                        help="Relative threshold density (w0 = ceil(rho * n_locs))")
    parser.add_argument("--w0-mode", choices=["volume", "expected", "tail"], default="volume",
                        help="Scale w0 by volume or expected weight under proposal.")
    parser.add_argument("--beta-prop", type=float, default=0.15)
    parser.add_argument("--beta-thresh", type=float, default=0.2)
    parser.add_argument("--filter-type", choices=["threshold", "softplus"], default="threshold")
    parser.add_argument("--softplus-tau", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-qiskit", action="store_true", help="Disable Qiskit simulation")
    args = parser.parse_args()

    print("=" * 60)
    print("THRESHOLD FILTER PEC VIA IMPORTANCE SAMPLING")
    print("=" * 60)

    t0 = time.time()
    data = benchmark(
        n_qubits=args.n_qubits,
        depth=args.depth,
        w0=args.w0,
        beta_prop=args.beta_prop,
        beta_thresh=args.beta_thresh,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        seed=args.seed,
        filter_type=args.filter_type,
        softplus_tau=args.softplus_tau,
        use_qiskit=not args.no_qiskit,
        w0_density=args.w0_density,
        w0_mode=args.w0_mode,
    )

    print_benchmark_results(data)
    print(f"\nTotal time: {time.time() - t0:.1f}s")

    filtered = data["results"]["filtered"]
    exp_results = data["results"]["exp_window"]
    full_results = data["results"]["full_pec"]

    filtered_rmse = np.sqrt(np.mean([d["error"] ** 2 for d in filtered]))
    exp_rmse = np.sqrt(np.mean([d["error"] ** 2 for d in exp_results]))
    full_rmse = np.sqrt(np.mean([d["error"] ** 2 for d in full_results]))

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)
    print(
        f"\nRMSE Comparison:\n"
        f"  Full PEC:     {full_rmse:.4f}\n"
        f"  Exponential:  {exp_rmse:.4f}  ({100 * (exp_rmse / full_rmse - 1):+.1f}% vs Full)\n"
        f"  Filtered:     {filtered_rmse:.4f}  ({100 * (filtered_rmse / full_rmse - 1):+.1f}% vs Full)\n"
    )


if __name__ == "__main__":
    main()
