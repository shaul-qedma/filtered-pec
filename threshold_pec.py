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

from rich import box
from rich.console import Console
from rich.table import Table

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

CLIP_MIN = -20.0
CLIP_MAX = 20.0
SOFTPLUS_TAU_DEFAULT = 1.0
MIN_SAMPLES = 1
MIN_STD_SAMPLES = 2
EPS_H_PROP = 1e-15
EPS_ESS = 1e-12
STD_DDOF = 1
SEED_OFFSET_FULL = 1000
SEED_OFFSET_EXP = 2000
SEED_OFFSET_FILTERED = 3000
QISKIT_METHOD = "statevector"
DEFAULT_N_QUBITS = 4
DEFAULT_DEPTH = 3
DEFAULT_BETA_PROP = 0.15
DEFAULT_BETA_THRESH = 0.2
DEFAULT_N_SAMPLES = 100
DEFAULT_N_TRIALS = 10
DEFAULT_SEED = 42
console = Console()

def h_threshold(weight: int, beta: float, w0: int) -> float:
    if weight <= w0:
        return 1.0
    return float(np.exp(-beta * (weight - w0)))


def h_softplus(weight: int, beta: float, w0: float, tau: float = SOFTPLUS_TAU_DEFAULT) -> float:
    x = (weight - w0) / tau
    x = float(np.clip(x, CLIP_MIN, CLIP_MAX))
    sp = np.log1p(np.exp(x)) if x <= CLIP_MAX else x
    return float(np.exp(-beta * sp * tau))


def critical_beta(p: np.ndarray) -> float:
    """Critical beta where local q becomes non-negative."""
    eigenvalues = ETA @ p
    min_eigenvalue = min(eigenvalues[1], eigenvalues[2], eigenvalues[3])
    if min_eigenvalue <= 0:
        return float("inf")
    return float(-np.log(min_eigenvalue))

def _resolve_w0(
    w0_density: Optional[float],
    n_locs: int,
) -> int:
    if w0_density is None:
        raise ValueError("threshold density must be provided")
    if w0_density < 0:
        raise ValueError("threshold density must be non-negative")
    scale = float(n_locs)
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
    if values.size < MIN_STD_SAMPLES:
        return 0.0
    return float(values.std(ddof=STD_DDOF)) / np.sqrt(values.size)


def _importance_sampling_estimate(
    measure_func: Callable[[Dict[Tuple[int, int], int]], float],
    error_locs: List[Tuple],
    beta_prop: float,
    h_target: Callable[[int], float],
    n_samples: int,
    rng: np.random.Generator,
    w0_value: Optional[int],
) -> ThresholdPECEstimate:
    if n_samples < MIN_SAMPLES:
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

        h_prop = float(np.exp(-beta_prop * weight))
        h_targ = h_target(weight)
        iw = h_targ / h_prop if h_prop > EPS_H_PROP else 0.0

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
    ess = n_samples / (1.0 + iw_var / (iw_mean ** 2)) if iw_mean > EPS_ESS else 0.0

    above_frac = float(np.mean(sample_weights > w0_value)) if w0_value is not None else float("nan")

    return ThresholdPECEstimate(
        mean=target_mean,
        std=target_std,
        qp_norm_proposal=total_qp_norm,
        effective_samples=float(ess),
        n_samples=n_samples,
        weight_mean=float(sample_weights.mean()),
        weight_std=float(sample_weights.std(ddof=STD_DDOF)) if n_samples >= MIN_STD_SAMPLES else 0.0,
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
    beta_prop: float,
    beta_thresh: float,
    n_samples: int,
    seed: int = 0,
    filter_type: str = "threshold",
    softplus_tau: float = SOFTPLUS_TAU_DEFAULT,
    h_target: Optional[Callable[[int], float]] = None,
    w0_density: float = 0.30,
) -> ThresholdPECEstimate:
    rng = np.random.default_rng(seed)
    w0_value = _resolve_w0(w0_density, len(error_locs))
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
    beta_prop: float,
    beta_thresh: float,
    n_samples: int,
    seed: int = 0,
    filter_type: str = "threshold",
    softplus_tau: float = SOFTPLUS_TAU_DEFAULT,
    h_target: Optional[Callable[[int], float]] = None,
    w0_density: float = 0.30,
) -> ThresholdPECEstimate:
    rng = np.random.default_rng(seed)
    w0_value = _resolve_w0(w0_density, len(error_locs))
    h_fn = _resolve_filter(filter_type, w0_value, beta_thresh, softplus_tau, h_target)

    QuantumCircuit, UnitaryGate, Operator, Pauli, Statevector, Kraus, AerSimulator = _require_qiskit()
    init_sv = _to_qiskit_statevector(initial_state, circuit.n_qubits)
    pauli_obs = Operator(Pauli(observable[::-1]))
    noise_instr = _qiskit_noise_instructions(noise, Kraus)
    backend = AerSimulator(method=QISKIT_METHOD)

    def measure(insertions: Dict[Tuple[int, int], int]) -> float:
        qc = _build_qiskit_circuit(
            circuit=circuit,
            init_statevector=init_sv,
            noise_instr=noise_instr,
            insertions=insertions,
            QuantumCircuit=QuantumCircuit,
            UnitaryGate=UnitaryGate,
        )
        qc.save_statevector()
        result = backend.run(qc, shots=1).result()
        state = result.data(0)["statevector"]
        sv = state if isinstance(state, Statevector) else Statevector(state)
        return float(np.real(sv.expectation_value(pauli_obs)))

    return _importance_sampling_estimate(
        measure, error_locs, beta_prop, h_fn, n_samples, rng, w0_value
    )


def benchmark(
    n_qubits: int,
    depth: int,
    beta_prop: float,
    beta_thresh: float,
    n_samples: int,
    n_trials: int,
    seed: int,
    filter_type: str = "threshold",
    softplus_tau: float = SOFTPLUS_TAU_DEFAULT,
    use_qiskit: bool = False,
    w0_density: float = 0.30,
) -> Dict:
    results = {
        "full_pec": [],
        "exp_window": [],
        "filtered": [],
    }
    ideals = []

    w0_values: List[int] = []
    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        circuit = random_circuit(n_qubits, depth, rng)
        noise = random_noise_model(rng)
        init = random_product_state(n_qubits, rng)
        obs = random_observable(n_qubits, rng)
        locs = error_locations(circuit, noise)
        w0_value = _resolve_w0(w0_density, len(locs))
        w0_values.append(w0_value)

        sim = NoisySimulator(STANDARD_GATES, noise)
        ideal = sim.ideal(circuit, obs, init)
        ideals.append(ideal)

        if use_qiskit:
            est_full = pec_estimate_qiskit(
                circuit,
                obs,
                init,
                noise,
                locs,
                beta=0.0,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_FULL * t,
            )
            est_exp = pec_estimate_qiskit(
                circuit,
                obs,
                init,
                noise,
                locs,
                beta=beta_prop,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_EXP * t,
            )
            est_filtered = threshold_pec_qiskit(
                circuit,
                obs,
                init,
                noise,
                locs,
                beta_prop=beta_prop,
                beta_thresh=beta_thresh,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_FILTERED * t,
                filter_type=filter_type,
                softplus_tau=softplus_tau,
                w0_density=w0_density,
            )
        else:
            est_full = pec_estimate(
                sim,
                circuit,
                obs,
                init,
                locs,
                beta=0.0,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_FULL * t,
            )
            est_exp = pec_estimate(
                sim,
                circuit,
                obs,
                init,
                locs,
                beta=beta_prop,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_EXP * t,
            )
            est_filtered = threshold_pec_estimate(
                sim,
                circuit,
                obs,
                init,
                locs,
                beta_prop=beta_prop,
                beta_thresh=beta_thresh,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_FILTERED * t,
                filter_type=filter_type,
                softplus_tau=softplus_tau,
                w0_density=w0_density,
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
            "w0_density": w0_density,
            "w0_mean": float(np.mean(w0_values)) if w0_values else 0.0,
            "w0_min": int(min(w0_values)) if w0_values else 0,
            "w0_max": int(max(w0_values)) if w0_values else 0,
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

    w0_label = (
        f"w0≈{config['w0_mean']:.1f} "
        f"(ρ={config['w0_density']}, range {config['w0_min']}-{config['w0_max']})"
    )

    console.rule("Threshold PEC Results")
    console.print(
        f"Config: {config['n_qubits']}q, depth={config['depth']}, {w0_label}, "
        f"β_prop={config['beta_prop']}, β_thresh={config['beta_thresh']}, filter={label}"
    )
    console.print(
        f"Samples: {config['n_samples']}, Trials: {config['n_trials']}, Qiskit={config['use_qiskit']}"
    )

    table = Table(box=box.ASCII)
    table.add_column("Method")
    table.add_column("qp_norm", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("ESS", justify="right")
    table.add_column("w_mean", justify="right")
    table.add_column("p_gt_w0", justify="right")

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
            table.add_row(
                name,
                f"{mean_qp_norm:.2f}",
                f"{bias:.4f}",
                f"{rmse:.4f}",
                f"{ess:.1f}",
                f"{w_mean:.2f}",
                f"{above:.2f}",
            )
        else:
            name = "full_pec" if method == "full_pec" else "exp_window"
            table.add_row(
                name,
                f"{mean_qp_norm:.2f}",
                f"{bias:.4f}",
                f"{rmse:.4f}",
                "N/A",
                "-",
                "-",
            )

    console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Threshold Filter PEC via Importance Sampling")
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--threshold-density", type=float, default=0.30,
                        help="Relative threshold density (w0 = ceil(rho * n_locs))")
    parser.add_argument("--beta-prop", type=float, default=DEFAULT_BETA_PROP)
    parser.add_argument("--beta-thresh", type=float, default=DEFAULT_BETA_THRESH)
    parser.add_argument("--filter-type", choices=["threshold", "softplus"], default="threshold")
    parser.add_argument("--softplus-tau", type=float, default=SOFTPLUS_TAU_DEFAULT)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-qiskit", action="store_true", help="Disable Qiskit simulation")
    args = parser.parse_args()

    console.rule("THRESHOLD FILTER PEC VIA IMPORTANCE SAMPLING")

    t0 = time.time()
    data = benchmark(
        n_qubits=args.n_qubits,
        depth=args.depth,
        beta_prop=args.beta_prop,
        beta_thresh=args.beta_thresh,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        seed=args.seed,
        filter_type=args.filter_type,
        softplus_tau=args.softplus_tau,
        use_qiskit=not args.no_qiskit,
        w0_density=args.threshold_density,
    )

    print_benchmark_results(data)
    console.print(f"Total time: {time.time() - t0:.1f}s")

    filtered = data["results"]["filtered"]
    exp_results = data["results"]["exp_window"]
    full_results = data["results"]["full_pec"]

    filtered_rmse = np.sqrt(np.mean([d["error"] ** 2 for d in filtered]))
    exp_rmse = np.sqrt(np.mean([d["error"] ** 2 for d in exp_results]))
    full_rmse = np.sqrt(np.mean([d["error"] ** 2 for d in full_results]))

    console.rule("ANALYSIS")
    table = Table(box=box.ASCII)
    table.add_column("Method")
    table.add_column("RMSE", justify="right")
    table.add_column("vs Full", justify="right")
    table.add_row("Full PEC", f"{full_rmse:.4f}", "baseline")
    table.add_row(
        "Exponential",
        f"{exp_rmse:.4f}",
        f"{100.0 * (exp_rmse / full_rmse - 1):+.1f}%",
    )
    table.add_row(
        "Filtered",
        f"{filtered_rmse:.4f}",
        f"{100.0 * (filtered_rmse / full_rmse - 1):+.1f}%",
    )
    console.print(table)


if __name__ == "__main__":
    main()
