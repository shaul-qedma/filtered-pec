"""
Parameter sweep for threshold/softplus PEC hyperparameters.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import yaml

from constants import DEFAULT_BATCH_SIZE, SOFTPLUS_TAU_DEFAULT
from threshold_pec import benchmark, generate_trials, TrialData

console = Console()
DEFAULT_CONFIG_SEED_OFFSET = 1000

_BATCH_SIZE_CACHE: Dict[Tuple[int, int], int] = {}


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a mapping.")
    
    # Validate noise_model if present in circuit_config
    circuit_config = data.get("circuit_config", {})
    if "noise_model" in circuit_config:
        nm = circuit_config["noise_model"]
        valid_types = {"kraus", "pauli", "per_gate", "depolarizing"}
        nm_type = nm.get("type")
        if nm_type not in valid_types:
            raise ValueError(f"noise_model.type must be one of {valid_types}, got '{nm_type}'")
        
        if nm_type == "per_gate":
            if nm.get("gates") is None:
                raise ValueError("per_gate type requires 'gates' configuration")
        elif nm_type == "depolarizing":
            if nm.get("infidelity") is None:
                raise ValueError("depolarizing type requires 'infidelity' field")
    
    return data


def _require_list(data: dict, key: str) -> List:
    value = data[key]
    if not isinstance(value, list):
        raise ValueError(f"Config field '{key}' must be a list.")
    return value


def _resolve_configs(config: dict) -> List[Tuple[int, int]]:
    if "configs" in config:
        raw_configs = _require_list(config, "configs")
        if not raw_configs:
            raise ValueError("configs must be a non-empty list.")
        resolved: List[Tuple[int, int]] = []
        for item in raw_configs:
            if isinstance(item, dict):
                n_qubits = item["n_qubits"]
                depth = item["depth"]
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                n_qubits, depth = item
            else:
                raise ValueError("Each config must be {n_qubits, depth} or [n_qubits, depth].")
            resolved.append((int(n_qubits), int(depth)))
        return resolved

    if "auto_configs" in config:
        raise ValueError("auto_configs is no longer supported. Use 'configs' instead.")

    if "n_qubits" in config or "depth" in config:
        if "n_qubits" not in config or "depth" not in config:
            raise ValueError("Both n_qubits and depth must be provided for a single config.")
        return [(int(config["n_qubits"]), int(config["depth"]))]

    raise ValueError("Config must define configs, or n_qubits/depth.")


def estimate_error_locs(n_qubits: int, depth: int) -> int:
    twoq_layers = depth // 2
    gates_per_layer = n_qubits // 2
    return 2 * gates_per_layer * twoq_layers


def _summary_from_filtered(results: List[dict]) -> dict:
    if not results:
        raise ValueError("Filtered results are empty.")
    errors = np.array([d["error"] for d in results], dtype=float)
    qp_norms = np.array([d["qp_norm"] for d in results], dtype=float)
    ess_vals = np.array([d["ess"] for d in results], dtype=float)
    above_vals = np.array([d["above"] for d in results], dtype=float)
    weight_vals = np.array([d["weight_mean"] for d in results], dtype=float)
    return {
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "qp_norm": float(np.mean(qp_norms)),
        "ess": float(np.mean(ess_vals)),
        "above": float(np.mean(above_vals)),
        "weight_mean": float(np.mean(weight_vals)),
    }


def _benchmark_count(
    w0_densities: List[float],
    beta_prop_values: List[float],
    beta_thresh_values: List[float],
    filter_type: str,
    softplus_taus: List[float],
) -> int:
    tau_count = len(softplus_taus) if filter_type == "softplus" else 1
    return 1 + len(w0_densities) * len(beta_prop_values) * len(beta_thresh_values) * tau_count


def _autotune_qiskit_batch_size(
    n_qubits: int,
    depth: int,
    trials: List[TrialData],
    n_samples: int,
    seed: int,
    beta_prop: float,
    beta_thresh: float,
    filter_type: str,
    softplus_tau: float,
    w0_density: float,
    candidates: List[int],
    tune_samples: int,
    tune_trials: int,
    warmup_repeats: int,
    timed_repeats: int,
    progress: bool = True,
) -> int:
    key = (n_qubits, depth)
    if key in _BATCH_SIZE_CACHE:
        return _BATCH_SIZE_CACHE[key]

    if not trials:
        raise ValueError("No trials available for batch-size autotune.")
    if not candidates:
        raise ValueError("autotune_candidates must be a non-empty list.")
    if tune_trials < 1:
        raise ValueError("autotune_trials must be at least 1.")
    if warmup_repeats < 0:
        raise ValueError("autotune_warmup must be >= 0.")
    if timed_repeats < 1:
        raise ValueError("autotune_repeats must be at least 1.")

    tune_samples = max(1, min(n_samples, tune_samples))
    batch_sizes = sorted({min(int(c), tune_samples) for c in candidates if int(c) > 0})
    if not batch_sizes:
        batch_sizes = [min(DEFAULT_BATCH_SIZE, tune_samples)]

    trial_count = max(1, min(len(trials), tune_trials))
    trial_subset = trials[:trial_count]
    best_size = batch_sizes[0]
    best_time = float("inf")
    
    batch_iter = tqdm(batch_sizes, desc="Autotuning batch size", disable=not progress)
    for batch_size in batch_iter:
        for warmup_idx in range(warmup_repeats):
            benchmark(
                n_qubits=n_qubits,
                depth=depth,
                beta_prop=beta_prop,
                beta_thresh=beta_thresh,
                n_samples=tune_samples,
                n_trials=len(trial_subset),
                seed=seed + 1000 + warmup_idx,
                filter_type=filter_type,
                softplus_tau=softplus_tau,
                w0_density=w0_density,
                trials=trial_subset,
                compute_full=False,
                compute_exp=False,
                compute_filtered=True,
                batch_size=batch_size,
                progress=False,
            )

        elapsed_runs = []
        for rep in range(timed_repeats):
            start = time.perf_counter()
            benchmark(
                n_qubits=n_qubits,
                depth=depth,
                beta_prop=beta_prop,
                beta_thresh=beta_thresh,
                n_samples=tune_samples,
                n_trials=len(trial_subset),
                seed=seed + 2000 + rep,
                filter_type=filter_type,
                softplus_tau=softplus_tau,
                w0_density=w0_density,
                trials=trial_subset,
                compute_full=False,
                compute_exp=False,
                compute_filtered=True,
                batch_size=batch_size,
                progress=False,
            )
            elapsed_runs.append(time.perf_counter() - start)
        median_time = float(np.median(np.array(elapsed_runs, dtype=float)))
        if median_time < best_time:
            best_time = median_time
            best_size = batch_size

    _BATCH_SIZE_CACHE[key] = best_size
    return best_size


def parameter_sweep(
    n_qubits: int,
    depth: int,
    n_samples: int,
    n_trials: int,
    seed: int,
    w0_densities: List[float],
    beta_prop_values: List[float],
    beta_thresh_values: List[float],
    filter_type: str,
    softplus_taus: List[float],
    trials: Optional[List[TrialData]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: bool = True,
    twoq_gates: Optional[List[str]] = None,
    noise_model_config: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Run parameter sweep over w0_density, beta_prop, beta_thresh, and optionally softplus_tau.
    """
    if trials is None:
        trials = generate_trials(
            n_qubits=n_qubits,
            depth=depth,
            n_trials=n_trials,
            seed=seed,
            twoq_gates=twoq_gates,
            noise_model_config=noise_model_config,
        )

    console.rule(f"Parameter Sweep: {n_qubits}q, depth={depth}")
    console.print(f"Samples: {n_samples}, Trials: {n_trials}, Batch size: {batch_size}")

    # Baseline: exponential window only
    baseline_data = benchmark(
        n_qubits=n_qubits,
        depth=depth,
        beta_prop=beta_prop_values[0],
        beta_thresh=0.0,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=seed,
        filter_type=filter_type,
        softplus_tau=softplus_taus[0] if softplus_taus else SOFTPLUS_TAU_DEFAULT,
        w0_density=w0_densities[0],
        trials=trials,
        compute_full=True,
        compute_exp=True,
        compute_filtered=False,
        batch_size=batch_size,
        progress=progress,
    )

    full_results = baseline_data["results"]["full_pec"]
    exp_results = baseline_data["results"]["exp_window"]
    
    if full_results:
        full_rmse = float(np.sqrt(np.mean([d["error"] ** 2 for d in full_results])))
        console.print(f"Full PEC RMSE: {full_rmse:.4f}")
    if exp_results:
        exp_rmse = float(np.sqrt(np.mean([d["error"] ** 2 for d in exp_results])))
        console.print(f"Exp Window RMSE: {exp_rmse:.4f}")

    # Sweep results
    sweep_results: List[Dict[str, Any]] = []
    
    tau_values = softplus_taus if filter_type == "softplus" else [SOFTPLUS_TAU_DEFAULT]
    
    total_combos = len(w0_densities) * len(beta_prop_values) * len(beta_thresh_values) * len(tau_values)
    combo_iter = tqdm(
        [(w0, bp, bt, tau) 
         for w0 in w0_densities 
         for bp in beta_prop_values 
         for bt in beta_thresh_values
         for tau in tau_values],
        desc="Sweep",
        total=total_combos,
        disable=not progress,
    )
    
    for w0_density, beta_prop, beta_thresh, tau in combo_iter:
        data = benchmark(
            n_qubits=n_qubits,
            depth=depth,
            beta_prop=beta_prop,
            beta_thresh=beta_thresh,
            n_samples=n_samples,
            n_trials=n_trials,
            seed=seed,
            filter_type=filter_type,
            softplus_tau=tau,
            w0_density=w0_density,
            trials=trials,
            compute_full=False,
            compute_exp=False,
            compute_filtered=True,
            batch_size=batch_size,
            progress=False,
        )
        
        stats = _summary_from_filtered(data["results"]["filtered"])
        sweep_results.append({
            "w0_density": w0_density,
            "beta_prop": beta_prop,
            "beta_thresh": beta_thresh,
            "softplus_tau": tau,
            **stats,
        })

    # Print results table
    table = Table(box=box.ASCII)
    table.add_column("ρ")
    table.add_column("β_prop")
    table.add_column("β_thresh")
    if filter_type == "softplus":
        table.add_column("τ")
    table.add_column("Bias", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_column("ESS", justify="right")
    table.add_column("qp_norm", justify="right")

    for r in sweep_results:
        row = [
            f"{r['w0_density']:.2f}",
            f"{r['beta_prop']:.2f}",
            f"{r['beta_thresh']:.2f}",
        ]
        if filter_type == "softplus":
            row.append(f"{r['softplus_tau']:.1f}")
        row.extend([
            f"{r['bias']:.4f}",
            f"{r['rmse']:.4f}",
            f"{r['ess']:.1f}",
            f"{r['qp_norm']:.1f}",
        ])
        table.add_row(*row)

    console.print(table)

    # Find best configuration
    best = min(sweep_results, key=lambda x: x["rmse"])
    console.print(f"\nBest config: ρ={best['w0_density']}, β_prop={best['beta_prop']}, "
                  f"β_thresh={best['beta_thresh']}, RMSE={best['rmse']:.4f}")

    return {
        "baseline": baseline_data,
        "sweep": sweep_results,
        "best": best,
    }


def compare_filters(
    n_qubits: int,
    depth: int,
    n_samples: int,
    n_trials: int,
    seed: int,
    w0_density: float,
    beta_prop: float,
    beta_thresh: float,
    softplus_taus: List[float],
    trials: Optional[List[TrialData]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: bool = True,
) -> None:
    """Compare threshold vs softplus filters."""
    console.rule("Filter Comparison: Threshold vs Softplus")

    if trials is None:
        trials = generate_trials(
            n_qubits=n_qubits,
            depth=depth,
            n_trials=n_trials,
            seed=seed,
        )

    # Threshold filter
    data_thresh = benchmark(
        n_qubits=n_qubits,
        depth=depth,
        beta_prop=beta_prop,
        beta_thresh=beta_thresh,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=seed,
        filter_type="threshold",
        w0_density=w0_density,
        trials=trials,
        compute_full=False,
        compute_exp=False,
        compute_filtered=True,
        batch_size=batch_size,
        progress=progress,
    )
    thresh_stats = _summary_from_filtered(data_thresh["results"]["filtered"])

    table = Table(box=box.ASCII)
    table.add_column("Filter")
    table.add_column("Bias", justify="right")
    table.add_column("RMSE", justify="right")

    table.add_row(
        "Threshold",
        f"{thresh_stats['bias']:.4f}",
        f"{thresh_stats['rmse']:.4f}",
    )

    # Softplus filters
    for tau in softplus_taus:
        data_soft = benchmark(
            n_qubits=n_qubits,
            depth=depth,
            beta_prop=beta_prop,
            beta_thresh=beta_thresh,
            n_samples=n_samples,
            n_trials=n_trials,
            seed=seed,
            filter_type="softplus",
            softplus_tau=tau,
            w0_density=w0_density,
            trials=trials,
            compute_full=False,
            compute_exp=False,
            compute_filtered=True,
            batch_size=batch_size,
            progress=progress,
        )
        soft_stats = _summary_from_filtered(data_soft["results"]["filtered"])
        table.add_row(
            f"Softplus τ={tau}",
            f"{soft_stats['bias']:.4f}",
            f"{soft_stats['rmse']:.4f}",
        )

    console.print(table)


def _run(config: dict) -> None:
    w0_densities = _require_list(config, "w0_densities")
    beta_prop_values = _require_list(config, "beta_props")
    beta_thresh_values = _require_list(config, "beta_threshes")
    softplus_taus = _require_list(config, "softplus_taus")
    filter_type = config["filter_type"]
    if filter_type not in {"threshold", "softplus"}:
        raise ValueError("filter_type must be 'threshold' or 'softplus'.")
    if not w0_densities:
        raise ValueError("At least one threshold density is required.")
    if not beta_prop_values:
        raise ValueError("At least one beta_prop value is required.")
    if not beta_thresh_values:
        raise ValueError("At least one beta_thresh value is required.")
    if filter_type == "softplus" and not softplus_taus:
        raise ValueError("softplus_taus cannot be empty for softplus sweeps.")

    n_samples = int(config["n_samples"])
    n_trials = int(config["n_trials"])
    seed = int(config["seed"])
    progress = config["progress"]
    skip_compare = config["skip_compare"]
    timings = config["timings"]
    profile = config["profile"]

    raw_batch = config["qiskit_batch_size"]
    autotune_batch = False
    if isinstance(raw_batch, str):
        if raw_batch.lower() != "auto":
            raise ValueError("qiskit_batch_size must be an int or 'auto'.")
        autotune_batch = True
        base_batch_size = DEFAULT_BATCH_SIZE
    else:
        base_batch_size = int(raw_batch)

    autotune_candidates = _require_list(config, "autotune_candidates")
    autotune_samples = int(config["autotune_samples"])
    autotune_trials = int(config["autotune_trials"])
    autotune_warmup = int(config["autotune_warmup"])
    autotune_repeats = int(config["autotune_repeats"])

    # Circuit generation configuration
    circuit_config = config.get("circuit_config", {})
    twoq_gates = circuit_config.get("twoq_gates")
    noise_model_config = circuit_config.get("noise_model")

    configs = _resolve_configs(config)
    benchmark_count = _benchmark_count(
        w0_densities=w0_densities,
        beta_prop_values=beta_prop_values,
        beta_thresh_values=beta_thresh_values,
        filter_type=filter_type,
        softplus_taus=softplus_taus,
    )

    timing_rows = []
    t0 = time.perf_counter()
    if not configs:
        raise ValueError("No configurations selected.")
    first_trials: Optional[List[TrialData]] = None
    first_config: Optional[Tuple[int, int]] = None
    first_batch_size: Optional[int] = None
    for idx, (n_qubits, depth) in enumerate(configs):
        if idx:
            console.rule()
        config_seed = seed + idx * DEFAULT_CONFIG_SEED_OFFSET
        
        if progress:
            console.print(f"[dim]Generating {n_trials} trials for {n_qubits}q, depth={depth}...[/dim]")
        
        trials = generate_trials(
            n_qubits=n_qubits,
            depth=depth,
            n_trials=n_trials,
            seed=config_seed,
            twoq_gates=twoq_gates,
            noise_model_config=noise_model_config,
            progress=progress,
        )
        if idx == 0:
            first_trials = trials
            first_config = (n_qubits, depth)

        batch_size = base_batch_size
        if autotune_batch:
            if progress:
                console.print(f"[dim]Autotuning batch size for {n_qubits}q, depth={depth}...[/dim]")
            tune_tau = softplus_taus[0] if filter_type == "softplus" else SOFTPLUS_TAU_DEFAULT
            batch_size = _autotune_qiskit_batch_size(
                n_qubits=n_qubits,
                depth=depth,
                trials=trials,
                n_samples=n_samples,
                seed=config_seed,
                beta_prop=beta_prop_values[0],
                beta_thresh=beta_thresh_values[0],
                filter_type=filter_type,
                softplus_tau=tune_tau,
                w0_density=w0_densities[0],
                candidates=autotune_candidates,
                tune_samples=autotune_samples,
                tune_trials=autotune_trials,
                warmup_repeats=autotune_warmup,
                timed_repeats=autotune_repeats,
                progress=progress,
            )
            if progress:
                console.print(f"Auto-tuned batch size: {batch_size}")

        if idx == 0:
            first_batch_size = batch_size

        config_start = time.perf_counter()
        parameter_sweep(
            n_qubits=n_qubits,
            depth=depth,
            n_samples=n_samples,
            n_trials=n_trials,
            seed=config_seed,
            w0_densities=w0_densities,
            beta_prop_values=beta_prop_values,
            beta_thresh_values=beta_thresh_values,
            filter_type=filter_type,
            softplus_taus=softplus_taus,
            trials=trials,
            batch_size=batch_size,
            progress=progress,
        )
        elapsed = time.perf_counter() - config_start
        if timings or profile:
            volume = estimate_error_locs(n_qubits, depth)
            timing_rows.append(
                {
                    "config": f"{n_qubits}q_d{depth}",
                    "volume": volume,
                    "benchmarks": benchmark_count,
                    "total_s": elapsed,
                }
            )

    if not skip_compare:
        if first_config is None or first_trials is None:
            raise ValueError("No configurations selected for comparison.")
        compare_n, compare_d = first_config
        compare_filters(
            n_qubits=compare_n,
            depth=compare_d,
            n_samples=n_samples,
            n_trials=n_trials,
            seed=seed + 1,
            w0_density=w0_densities[0],
            beta_prop=beta_prop_values[0],
            beta_thresh=beta_thresh_values[0],
            softplus_taus=softplus_taus,
            trials=first_trials,
            batch_size=first_batch_size or base_batch_size,
            progress=progress,
        )

    if timing_rows:
        table = Table(box=box.ASCII)
        table.add_column("Config")
        table.add_column("Volume", justify="right")
        table.add_column("Benchmarks", justify="right")
        table.add_column("Total (s)", justify="right")
        table.add_column("s/bench", justify="right")
        table.add_column("ms/trial", justify="right")
        table.add_column("µs/sample", justify="right")
        for row in timing_rows:
            per_bench = row["total_s"] / row["benchmarks"]
            per_trial = per_bench / n_trials
            per_sample = per_trial / n_samples
            table.add_row(
                row["config"],
                f"{row['volume']}",
                f"{row['benchmarks']}",
                f"{row['total_s']:.2f}",
                f"{per_bench:.3f}",
                f"{per_trial * 1000.0:.2f}",
                f"{per_sample * 1_000_000.0:.2f}",
            )
        console.rule("Timing Summary")
        console.print(table)

    console.print(f"Total time: {time.perf_counter() - t0:.1f}s")


def _run_with_profile(config: dict) -> None:
    profile_out = str(config["profile_out"])
    profile_top = int(config["profile_top"])
    profile_sort = str(config["profile_sort"])

    profiler = cProfile.Profile()
    profiler.enable()
    _run(config)
    profiler.disable()

    profiler.dump_stats(profile_out)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(profile_sort)
    stats.print_stats(profile_top)
    console.rule("Profile Summary")
    console.print(stream.getvalue())
    console.print(f"Profile saved to {profile_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep threshold/softplus PEC from a YAML config.")
    parser.add_argument("config", type=str, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = _load_config(args.config)
    if config["profile"]:
        _run_with_profile(config)
    else:
        _run(config)


if __name__ == "__main__":
    main()