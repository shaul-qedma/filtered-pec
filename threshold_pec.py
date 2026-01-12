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
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from backend import Backend, QiskitStatevector
from constants import (
    CLIP_MIN, CLIP_MAX, SOFTPLUS_TAU_DEFAULT,
    MIN_SAMPLES, EPS_H_PROP, EPS_ESS, STD_DDOF,
    SEED_OFFSET_FULL, SEED_OFFSET_EXP, SEED_OFFSET_FILTERED,
    DEFAULT_N_QUBITS, DEFAULT_DEPTH, DEFAULT_BETA_PROP, DEFAULT_BETA_THRESH,
    DEFAULT_N_SAMPLES, DEFAULT_N_TRIALS, DEFAULT_SEED,
    MIN_STD_SAMPLES, DEFAULT_BATCH_SIZE,
)
from diagnostics import (
    DEFAULT_DIAGNOSTICS,
    estimate_bias_from_samples,
    decompose_variance,
    analytical_bias_bound,
    weight_conditioned_variance,
    baseline_tracking,
)
from estimators import PECEstimate, ThresholdPECEstimate
from exp_window_pec import exp_window_quasi_prob, qp_norm, pec_estimate
from noise_model import generate_noise_model
from pec_shared import (
    STANDARD_GATES,
    CLIFFORD_2Q_GATES,
    Circuit,
    error_locations,
    random_circuit,
    random_basis_state,
    random_product_state,
    random_observable,
)

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
class TrialData:
    circuit: Circuit
    noise: dict
    initial_state: np.ndarray
    observable: str
    error_locs: List[Tuple]
    ideal: float


def _compute_ideal(
    circuit: Circuit,
    observable: str,
    initial_state: np.ndarray,
    circuit_type: str,
) -> float:
    """Compute ideal expectation value (noiseless, no insertions)."""
    if circuit_type == "clifford":
        from clifford import stabilizer_expectation

        return stabilizer_expectation(circuit, observable, initial_state)

    if circuit_type != "brickwall":
        raise ValueError(f"Unknown circuit_type: {circuit_type}")

    backend = QiskitStatevector()
    empty_noise: dict = {}
    return backend.expectation(circuit, observable, initial_state, empty_noise, {})


def generate_trials(
    n_qubits: int,
    depth: int,
    n_trials: int,
    seed: int,
    twoq_gates: Optional[List[str]] = None,
    noise_model_config: Optional[Dict] = None,
    progress: bool = False,
    circuit_type: str = "brickwall",
) -> List[TrialData]:
    trials: List[TrialData] = []
    trial_iter = tqdm(range(n_trials), desc="Generating trials", disable=not progress)
    for t in trial_iter:
        rng = np.random.default_rng(seed + t)
        noise_seed = None
        if noise_model_config is not None and "seed" in noise_model_config:
            noise_seed = int(noise_model_config["seed"]) + t
        noise_rng = np.random.default_rng(noise_seed) if noise_seed is not None else rng
        if circuit_type == "clifford":
            from clifford import random_clifford_circuit

            circuit = random_clifford_circuit(n_qubits, depth, rng, twoq_gates)
            init = random_basis_state(n_qubits, rng)
        elif circuit_type == "brickwall":
            circuit = random_circuit(n_qubits, depth, rng, twoq_gates)
            init = random_product_state(n_qubits, rng)
        else:
            raise ValueError(f"Unknown circuit_type: {circuit_type}")
        noise = generate_noise_model(noise_rng, noise_model_config, twoq_gates)
        obs = random_observable(n_qubits, rng)
        locs = error_locations(circuit, noise)
        ideal = _compute_ideal(circuit, obs, init, circuit_type)
        trials.append(
            TrialData(
                circuit=circuit,
                noise=noise,
                initial_state=init,
                observable=obs,
                error_locs=locs,
                ideal=ideal,
            )
        )
    return trials


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


def _importance_sampling_estimate(
    measure_func: Callable[[Dict[Tuple[int, int], int]], float],
    error_locs: List[Tuple],
    beta_prop: float,
    beta_thresh: float,
    h_target: Callable[[int], float],
    n_samples: int,
    rng: np.random.Generator,
    w0_value: Optional[int],
    diagnostics: Optional[Dict[str, bool]] = None,
    progress: bool = False,
    measure_batch_func: Optional[Callable[[List[Dict[Tuple[int, int], int]]], List[float]]] = None,
    batch_size: int = 0,
) -> ThresholdPECEstimate:
    if n_samples < MIN_SAMPLES:
        raise ValueError("n_samples must be positive")
    diagnostics = DEFAULT_DIAGNOSTICS if diagnostics is None else diagnostics

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
        )

    local_q = [exp_window_quasi_prob(p, beta_prop) for (_, _, p) in error_locs]
    local_qp_norm = [qp_norm(q) for q in local_q]
    total_qp_norm = float(np.prod(local_qp_norm)) if local_qp_norm else 1.0

    sampling_probs = [np.abs(q) / g for q, g in zip(local_q, local_qp_norm)]
    sampling_signs = [np.sign(q) for q in local_q]

    target_estimates = np.empty(n_samples, dtype=float)
    importance_weights = np.empty(n_samples, dtype=float)
    sample_weights = np.empty(n_samples, dtype=float)
    record_measurements = diagnostics["bias_gamma"]
    raw_measurements = np.empty(n_samples, dtype=float) if record_measurements else None
    batch_indices = np.zeros(n_samples, dtype=int) if diagnostics["baseline_tracking"] else None

    show_progress = progress and sys.stderr.isatty()
    if measure_batch_func is None:
        sample_iter = range(n_samples)
        if show_progress:
            sample_iter = tqdm(
                sample_iter,
                desc="Samples",
                unit="sample",
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
        for i in sample_iter:
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
            if raw_measurements is not None:
                raw_measurements[i] = measurement

            h_prop = float(np.exp(-beta_prop * weight))
            h_targ = h_target(weight)
            iw = h_targ / h_prop if h_prop > EPS_H_PROP else 0.0

            target_estimates[i] = iw * exp_val
            importance_weights[i] = iw
            sample_weights[i] = weight
    else:
        batch_size = n_samples if batch_size <= 0 else batch_size
        batch_id = 0
        total_bar = None
        if show_progress:
            total_bar = tqdm(
                total=n_samples,
                desc="Samples",
                unit="sample",
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
        for start in range(0, n_samples, batch_size):
            end = min(n_samples, start + batch_size)
            insertions_list: List[Dict[Tuple[int, int], int]] = []
            signs = np.empty(end - start, dtype=float)
            weights = np.empty(end - start, dtype=float)
            for i in range(start, end):
                insertions: Dict[Tuple[int, int], int] = {}
                sign = 1.0
                weight = 0
                for v, (layer, qubit, _) in enumerate(error_locs):
                    s = int(rng.choice(4, p=sampling_probs[v]))
                    sign *= sampling_signs[v][s]
                    if s != 0:
                        insertions[(layer, qubit)] = s
                        weight += 1
                insertions_list.append(insertions)
                signs[i - start] = sign
                weights[i - start] = weight

            measurements = measure_batch_func(insertions_list)
            if raw_measurements is not None:
                raw_measurements[start:end] = np.array(measurements, dtype=float)
            if batch_indices is not None:
                batch_indices[start:end] = batch_id
            for j, measurement in enumerate(measurements):
                exp_val = signs[j] * float(measurement)
                h_prop = float(np.exp(-beta_prop * weights[j]))
                h_targ = h_target(int(weights[j]))
                iw = h_targ / h_prop if h_prop > EPS_H_PROP else 0.0
                target_estimates[start + j] = iw * exp_val
                importance_weights[start + j] = iw
                sample_weights[start + j] = weights[j]
            batch_id += 1
            if total_bar is not None:
                total_bar.update(end - start)
        if total_bar is not None:
            total_bar.close()

    target_mean = total_qp_norm * float(target_estimates.mean())
    target_std = total_qp_norm * (target_estimates.std(ddof=STD_DDOF) / np.sqrt(n_samples) if n_samples >= MIN_STD_SAMPLES else 0.0)

    iw_mean = float(np.mean(importance_weights))
    iw_var = float(np.var(importance_weights))
    ess = n_samples / (1.0 + iw_var / (iw_mean ** 2)) if iw_mean > EPS_ESS else 0.0

    above_frac = float(np.mean(sample_weights > w0_value)) if w0_value is not None else float("nan")

    bias_bound_gamma = float("nan")
    gamma_diff = float("nan")
    gamma_std = float("nan")
    if diagnostics["bias_gamma"] and raw_measurements is not None and w0_value is not None:
        bias_bound_gamma, gamma_diff, gamma_std = estimate_bias_from_samples(
            raw_measurements,
            sample_weights,
            w0_value,
        )

    bias_bound_analytical = float("nan")
    if diagnostics["bias_analytical"]:
        bias_bound_analytical = analytical_bias_bound(error_locs, beta_thresh)

    var_total = float("nan")
    var_circuit = float("nan")
    var_shot = float("nan")
    if diagnostics["variance_decomp"]:
        var_total, var_circuit, var_shot = decompose_variance(target_estimates)

    weight_variance = None
    if diagnostics["weight_variance"]:
        weight_variance = weight_conditioned_variance(target_estimates, sample_weights, importance_weights)

    baseline_mean = float("nan")
    baseline_std = float("nan")
    baseline_count = 0
    baseline_drift = False
    baseline_z = float("nan")
    if diagnostics["baseline_tracking"]:
        (
            baseline_mean,
            baseline_std,
            baseline_count,
            baseline_drift,
            baseline_z,
        ) = baseline_tracking(target_estimates, sample_weights, batch_indices)

    return ThresholdPECEstimate(
        mean=target_mean,
        std=target_std,
        qp_norm_proposal=total_qp_norm,
        effective_samples=float(ess),
        n_samples=n_samples,
        weight_mean=float(sample_weights.mean()),
        weight_std=float(sample_weights.std(ddof=STD_DDOF)) if n_samples >= MIN_STD_SAMPLES else 0.0,
        weight_above_frac=above_frac,
        bias_bound_gamma=bias_bound_gamma,
        gamma_diff=gamma_diff,
        gamma_std=gamma_std,
        bias_bound_analytical=bias_bound_analytical,
        var_total=var_total,
        var_circuit=var_circuit,
        var_shot=var_shot,
        weight_variance=weight_variance,
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        baseline_count=baseline_count,
        baseline_drift=baseline_drift,
        baseline_z=baseline_z,
    )


def threshold_pec_estimate(
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
    backend: Backend | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    diagnostics: Optional[Dict[str, bool]] = None,
    progress: bool = False,
) -> ThresholdPECEstimate:
    """
    Threshold-filtered PEC estimation.
    
    Args:
        circuit: Quantum circuit
        observable: Pauli string (e.g., "XZI")
        initial_state: Initial state vector
        noise: Noise model dict
        error_locs: List of (layer, qubit, noise_probs) tuples
        beta_prop: Proposal distribution suppression parameter
        beta_thresh: Target filter suppression parameter
        n_samples: Number of Monte Carlo samples
        seed: Random seed
        filter_type: "threshold" or "softplus"
        softplus_tau: Softplus temperature parameter
        h_target: Custom target filter function (overrides filter_type)
        w0_density: Threshold as fraction of error locations
        backend: Simulation backend (default: QiskitStatevector)
        batch_size: Batch size for circuit execution
        diagnostics: Dict of diagnostic flags
    
    Returns:
        ThresholdPECEstimate with mean, std, diagnostics
    """
    if backend is None:
        backend = QiskitStatevector(batch_size=batch_size)
    
    rng = np.random.default_rng(seed)
    w0_value = _resolve_w0(w0_density, len(error_locs))
    h_fn = _resolve_filter(filter_type, w0_value, beta_thresh, softplus_tau, h_target)

    def measure(insertions: Dict[Tuple[int, int], int]) -> float:
        return backend.expectation(circuit, observable, initial_state, noise, insertions)

    def measure_batch(insertions_list: List[Dict[Tuple[int, int], int]]) -> List[float]:
        return backend.batch_expectations(circuit, observable, initial_state, noise, insertions_list)

    return _importance_sampling_estimate(
        measure,
        error_locs,
        beta_prop,
        beta_thresh,
        h_fn,
        n_samples,
        rng,
        w0_value,
        diagnostics=diagnostics,
        progress=progress,
        measure_batch_func=measure_batch,
        batch_size=batch_size,
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
    w0_density: float = 0.30,
    trials: Optional[List[TrialData]] = None,
    compute_full: bool = True,
    compute_exp: bool = True,
    compute_filtered: bool = True,
    backend: Backend | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: bool = True,
    diagnostics: Optional[Dict[str, bool]] = None,
    twoq_gates: Optional[List[str]] = None,
    noise_model_config: Optional[Dict] = None,
    circuit_type: str = "brickwall",
) -> Dict:
    """
    Run PEC benchmark comparing full, exponential window, and threshold-filtered methods.
    """
    if backend is None:
        backend = QiskitStatevector(batch_size=batch_size)
    
    if diagnostics is None:
        diagnostics = DEFAULT_DIAGNOSTICS.copy()

    if trials is None:
        trials = generate_trials(
            n_qubits=n_qubits,
            depth=depth,
            n_trials=n_trials,
            seed=seed,
            twoq_gates=twoq_gates,
            noise_model_config=noise_model_config,
            circuit_type=circuit_type,
        )

    results: Dict[str, List[Dict[str, Any]]] = {
        "full_pec": [],
        "exp_window": [],
        "filtered": [],
    }
    ideals: List[float] = []
    w0_values: List[int] = []

    trial_iter = tqdm(trials, desc="Trials", disable=not progress)
    for t, trial in enumerate(trial_iter):
        circuit = trial.circuit
        noise = trial.noise
        init = trial.initial_state
        obs = trial.observable
        locs = trial.error_locs
        ideal = trial.ideal
        ideals.append(ideal)

        w0_value = _resolve_w0(w0_density, len(locs))
        w0_values.append(w0_value)

        est_full: Optional[PECEstimate] = None
        est_exp: Optional[PECEstimate] = None
        est_filtered: Optional[ThresholdPECEstimate] = None

        if compute_full:
            est_full = pec_estimate(
                circuit=circuit,
                observable=obs,
                initial_state=init,
                noise=noise,
                error_locs=locs,
                beta=0.0,  # Full PEC
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_FULL + t,
                backend=backend,
                batch_size=batch_size,
                progress=progress,
            )

        if compute_exp:
            est_exp = pec_estimate(
                circuit=circuit,
                observable=obs,
                initial_state=init,
                noise=noise,
                error_locs=locs,
                beta=beta_prop,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_EXP + t,
                backend=backend,
                batch_size=batch_size,
                progress=progress,
            )

        if compute_filtered:
            est_filtered = threshold_pec_estimate(
                circuit=circuit,
                observable=obs,
                initial_state=init,
                noise=noise,
                error_locs=locs,
                beta_prop=beta_prop,
                beta_thresh=beta_thresh,
                n_samples=n_samples,
                seed=seed + SEED_OFFSET_FILTERED + t,
                filter_type=filter_type,
                softplus_tau=softplus_tau,
                w0_density=w0_density,
                backend=backend,
                batch_size=batch_size,
                diagnostics=diagnostics,
                progress=progress,
            )

        if compute_full and est_full is not None:
            results["full_pec"].append(
                {"estimate": est_full.mean, "qp_norm": est_full.qp_norm, "error": est_full.mean - ideal}
            )
        if compute_exp and est_exp is not None:
            results["exp_window"].append(
                {"estimate": est_exp.mean, "qp_norm": est_exp.qp_norm, "error": est_exp.mean - ideal}
            )
        if compute_filtered and est_filtered is not None:
            filtered_row: Dict[str, Any] = {
                "estimate": est_filtered.mean,
                "qp_norm": est_filtered.qp_norm_proposal,
                "ess": est_filtered.effective_samples,
                "above": est_filtered.weight_above_frac,
                "weight_mean": est_filtered.weight_mean,
                "weight_std": est_filtered.weight_std,
                "error": est_filtered.mean - ideal,
            }
            if diagnostics["bias_gamma"]:
                filtered_row["bias_bound_gamma"] = est_filtered.bias_bound_gamma
                filtered_row["gamma_diff"] = est_filtered.gamma_diff
                filtered_row["gamma_std"] = est_filtered.gamma_std
            if diagnostics["variance_decomp"]:
                filtered_row["var_total"] = est_filtered.var_total
                filtered_row["var_circuit"] = est_filtered.var_circuit
                filtered_row["var_shot"] = est_filtered.var_shot
            if diagnostics["bias_analytical"]:
                filtered_row["bias_bound_analytical"] = est_filtered.bias_bound_analytical
            if diagnostics["weight_variance"]:
                filtered_row["weight_variance"] = est_filtered.weight_variance
            if diagnostics["baseline_tracking"]:
                filtered_row["baseline_mean"] = est_filtered.baseline_mean
                filtered_row["baseline_std"] = est_filtered.baseline_std
                filtered_row["baseline_count"] = est_filtered.baseline_count
                filtered_row["baseline_drift"] = est_filtered.baseline_drift
                filtered_row["baseline_z"] = est_filtered.baseline_z
            results["filtered"].append(filtered_row)

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
            "n_trials": len(trials),
            "filter_type": filter_type,
            "softplus_tau": softplus_tau,
            "circuit_type": circuit_type,
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
        f"Samples: {config['n_samples']}, Trials: {config['n_trials']}"
    )

    table = Table(box=box.ASCII)
    table.add_column("Method")
    table.add_column("qp_norm", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("ESS", justify="right")

    for method in ["full_pec", "exp_window", "filtered"]:
        data_m = results[method]
        if not data_m:
            continue
        errors = np.array([d["error"] for d in data_m])
        qp_norms = np.array([d["qp_norm"] for d in data_m])

        bias = float(np.mean(errors))
        rmse = float(np.sqrt(np.mean(errors**2)))
        mean_qp_norm = float(np.mean(qp_norms))

        if method == "filtered":
            ess = float(np.mean([d["ess"] for d in data_m]))
            name = label
            table.add_row(
                name,
                f"{mean_qp_norm:.2f}",
                f"{bias:.4f}",
                f"{rmse:.4f}",
                f"{ess:.1f}",
            )
        else:
            name = "full_pec" if method == "full_pec" else "exp_window"
            table.add_row(
                name,
                f"{mean_qp_norm:.2f}",
                f"{bias:.4f}",
                f"{rmse:.4f}",
                "N/A",
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
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
        w0_density=args.threshold_density,
        batch_size=args.batch_size,
    )

    print_benchmark_results(data)
    console.print(f"Total time: {time.time() - t0:.1f}s")

    filtered = data["results"]["filtered"]
    exp_results = data["results"]["exp_window"]
    full_results = data["results"]["full_pec"]

    if filtered and exp_results and full_results:
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
