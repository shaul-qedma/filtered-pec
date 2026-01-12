"""
Parameter sweep for threshold/softplus PEC hyperparameters.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table
from tqdm import tqdm
import yaml

from exp_window_pec import DEFAULT_QISKIT_BATCH_SIZE
from threshold_pec import benchmark, generate_trials, TrialData

console = Console()
DEFAULT_SOFTPLUS_TAU = 1.0
DEFAULT_CONFIG_SEED_OFFSET = 1000

_BATCH_SIZE_CACHE: Dict[Tuple[int, int], int] = {}

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Sweep config must be a mapping.")
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
        auto = config["auto_configs"]
        if not isinstance(auto, dict):
            raise ValueError("auto_configs must be a mapping.")
        qubits = _require_list(auto, "qubits")
        depths = _require_list(auto, "depths")
        ratios = _require_list(auto, "ratios")
        min_volume = int(auto["min_volume"])
        max_volume = int(auto["max_volume"])
        per_ratio = int(auto["per_ratio"])
        max_configs = int(auto["max_configs"])
        return auto_configs(
            qubits=qubits,
            depths=depths,
            min_volume=min_volume,
            max_volume=max_volume if max_volume > 0 else None,
            ratios=ratios,
            per_ratio=max(1, per_ratio),
            max_configs=max(0, max_configs),
        )

    if "n_qubits" in config or "depth" in config:
        if "n_qubits" not in config or "depth" not in config:
            raise ValueError("Both n_qubits and depth must be provided for a single config.")
        return [(int(config["n_qubits"]), int(config["depth"]))]

    raise ValueError("Config must define configs, auto_configs, or n_qubits/depth.")


def estimate_error_locs(n_qubits: int, depth: int) -> int:
    twoq_layers = depth // 2
    gates_per_layer = n_qubits // 2
    return 2 * gates_per_layer * twoq_layers


def auto_configs(
    qubits: List[int],
    depths: List[int],
    min_volume: int,
    max_volume: Optional[int],
    ratios: List[float],
    per_ratio: int,
    max_configs: int,
) -> List[Tuple[int, int]]:
    candidates: List[Tuple[int, int, int, float]] = []
    for n in qubits:
        for d in depths:
            volume = estimate_error_locs(n, d)
            if volume < min_volume:
                continue
            if max_volume is not None and volume > max_volume:
                continue
            ratio = d / n if n else 0.0
            candidates.append((n, d, volume, ratio))

    if not candidates:
        raise ValueError("No configurations match the volume constraints.")

    volume_mid = min_volume
    if max_volume is not None:
        volume_mid = 0.5 * (min_volume + max_volume)

    chosen: List[Tuple[int, int]] = []
    used = set()

    if ratios:
        for r in ratios:
            ranked = sorted(
                candidates,
                key=lambda c: (abs(c[3] - r), abs(c[2] - volume_mid), c[2]),
            )
            picked = 0
            for n, d, _, _ in ranked:
                if (n, d) in used:
                    continue
                chosen.append((n, d))
                used.add((n, d))
                picked += 1
                if picked >= per_ratio:
                    break

    if not chosen:
        ranked = sorted(candidates, key=lambda c: (c[2], c[3]))
        chosen = [(n, d) for n, d, _, _ in ranked]

    if max_configs > 0 and len(chosen) > max_configs:
        chosen = chosen[:max_configs]

    return chosen


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
        batch_sizes = [min(DEFAULT_QISKIT_BATCH_SIZE, tune_samples)]

    trial_count = max(1, min(len(trials), tune_trials))
    trial_subset = trials[:trial_count]
    best_size = batch_sizes[0]
    best_time = float("inf")
    for batch_size in batch_sizes:
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
                use_qiskit=True,
                w0_density=w0_density,
                trials=trial_subset,
                compute_full=False,
                compute_exp=False,
                compute_filtered=True,
                qiskit_batch_size=batch_size,
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
                use_qiskit=True,
                w0_density=w0_density,
                trials=trial_subset,
                compute_full=False,
                compute_exp=False,
                compute_filtered=True,
                qiskit_batch_size=batch_size,
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
    use_qiskit: bool,
    trials: List[TrialData],
    qiskit_batch_size: int,
    progress: bool,
) -> List[dict]:
    console.rule("THRESHOLD PEC PARAMETER SWEEP")

    volume = estimate_error_locs(n_qubits, depth)
    console.print(f"Circuit: {n_qubits} qubits, depth {depth}, volume={volume}")
    console.print(f"Samples: {n_samples}, Trials: {len(trials)}, Qiskit={use_qiskit}")

    benchmark_total = _benchmark_count(
        w0_densities=w0_densities,
        beta_prop_values=beta_prop_values,
        beta_thresh_values=beta_thresh_values,
        filter_type=filter_type,
        softplus_taus=softplus_taus,
    )
    show_progress = progress and sys.stderr.isatty()
    with tqdm(
        total=benchmark_total,
        desc=f"Sweep {n_qubits}q d{depth}",
        unit="bench",
        dynamic_ncols=True,
        leave=False,
        disable=not show_progress,
    ) as bench_bar:
        data_full = benchmark(
            n_qubits=n_qubits,
            depth=depth,
            beta_prop=0.0,
            beta_thresh=0.0,
            n_samples=n_samples,
            n_trials=n_trials,
            seed=seed,
            filter_type="threshold",
            softplus_tau=DEFAULT_SOFTPLUS_TAU,
            use_qiskit=use_qiskit,
            w0_density=w0_densities[0],
            trials=trials,
            compute_full=True,
            compute_exp=False,
            compute_filtered=False,
            qiskit_batch_size=qiskit_batch_size,
            progress=False,
        )
        bench_bar.update(1)
        full_errors = np.array([d["error"] for d in data_full["results"]["full_pec"]])
        full_rmse = float(np.sqrt(np.mean(full_errors**2)))
        full_qp_norm = float(
            np.mean(np.array([d["qp_norm"] for d in data_full["results"]["full_pec"]], dtype=float))
        )
        full_bias = float(np.mean(full_errors))

        results = []

        table = Table(box=box.ASCII)
        table.add_column("Method")
        table.add_column("ρ", justify="right")
        table.add_column("w0", justify="right")
        table.add_column("β_prop", justify="right")
        table.add_column("β_thresh", justify="right")
        if filter_type == "softplus":
            table.add_column("τ", justify="right")
        table.add_column("qp_norm", justify="right")
        table.add_column("Bias", justify="right")
        table.add_column("RMSE", justify="right")
        table.add_column("ESS", justify="right")
        table.add_column("vs Full", justify="right")

        full_row = ["Full", "-"]
        full_row.extend(["-", "-", "-"])
        if filter_type == "softplus":
            full_row.append("-")
        full_row.extend(
            [
                f"{full_qp_norm:.2f}",
                f"{full_bias:.4f}",
                f"{full_rmse:.4f}",
                "-",
                "baseline",
            ]
        )
        table.add_row(*full_row)

        for rho in w0_densities:
            for beta_prop in beta_prop_values:
                for beta_thresh in beta_thresh_values:
                    taus = softplus_taus if filter_type == "softplus" else [DEFAULT_SOFTPLUS_TAU]
                    for tau in taus:
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
                            use_qiskit=use_qiskit,
                            w0_density=rho,
                            trials=trials,
                            compute_full=False,
                            compute_exp=False,
                            compute_filtered=True,
                            qiskit_batch_size=qiskit_batch_size,
                            progress=False,
                        )
                        bench_bar.update(1)

                        stats = _summary_from_filtered(data["results"]["filtered"])
                        vs_full = 100.0 * (stats["rmse"] / full_rmse - 1)

                        row = {
                            "w0_density": rho,
                            "beta_prop": beta_prop,
                            "beta_thresh": beta_thresh,
                            "tau": tau,
                            "qp_norm": stats["qp_norm"],
                            "bias": stats["bias"],
                            "rmse": stats["rmse"],
                            "ess": stats["ess"],
                            "vs_full": vs_full,
                            "w0_mean": data["config"]["w0_mean"],
                        }
                        results.append(row)

                        row = ["softplus" if filter_type == "softplus" else "threshold", f"{rho:.2f}"]
                        row.append(f"{data['config']['w0_mean']:.1f}")
                        row.append(f"{beta_prop:.2f}")
                        row.append(f"{beta_thresh:.2f}")
                        if filter_type == "softplus":
                            row.append(f"{tau:.2f}")
                        row.extend(
                            [
                                f"{stats['qp_norm']:.2f}",
                                f"{stats['bias']:.4f}",
                                f"{stats['rmse']:.4f}",
                                f"{stats['ess']:.1f}",
                                f"{vs_full:+.1f}%",
                            ]
                        )
                        table.add_row(*row)

    console.print(table)
    best = min(results, key=lambda r: r["rmse"])
    console.print("Best configuration:")
    if filter_type == "softplus":
        console.print(
            f"  ρ={best['w0_density']}, w0≈{best['w0_mean']:.1f}, "
            f"β_prop={best['beta_prop']}, β_thresh={best['beta_thresh']}, τ={best['tau']}"
        )
    else:
        console.print(
            f"  ρ={best['w0_density']}, w0≈{best['w0_mean']:.1f}, "
            f"β_prop={best['beta_prop']}, β_thresh={best['beta_thresh']}"
        )
    console.print(f"  RMSE = {best['rmse']:.4f} ({best['vs_full']:+.1f}% vs Full PEC)")

    return results


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
    use_qiskit: bool,
    trials: Optional[List[TrialData]] = None,
    qiskit_batch_size: int = DEFAULT_QISKIT_BATCH_SIZE,
    progress: bool = False,
) -> None:
    console.rule("THRESHOLD vs SOFTPLUS FILTER COMPARISON")

    data_thresh = benchmark(
        n_qubits=n_qubits,
        depth=depth,
        beta_prop=beta_prop,
        beta_thresh=beta_thresh,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=seed,
        filter_type="threshold",
        softplus_tau=DEFAULT_SOFTPLUS_TAU,
        use_qiskit=use_qiskit,
        w0_density=w0_density,
        trials=trials,
        qiskit_batch_size=qiskit_batch_size,
        progress=progress,
    )
    thresh_stats = _summary_from_filtered(data_thresh["results"]["filtered"])

    w0_label = f"ρ={w0_density}, w0≈{data_thresh['config']['w0_mean']:.1f}"

    console.print(f"Config: {n_qubits}q, depth={depth}, {w0_label}, β_prop={beta_prop}, β_thresh={beta_thresh}")
    table = Table(box=box.ASCII)
    table.add_column("Filter")
    table.add_column("Bias", justify="right")
    table.add_column("RMSE", justify="right")
    table.add_row(
        "Threshold",
        f"{thresh_stats['bias']:.4f}",
        f"{thresh_stats['rmse']:.4f}",
    )

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
            use_qiskit=use_qiskit,
            w0_density=w0_density,
            trials=trials,
            qiskit_batch_size=qiskit_batch_size,
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
    use_qiskit = config["use_qiskit"]
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
        base_batch_size = DEFAULT_QISKIT_BATCH_SIZE
    else:
        base_batch_size = int(raw_batch)

    autotune_candidates = _require_list(config, "autotune_candidates")
    autotune_samples = int(config["autotune_samples"])
    autotune_trials = int(config["autotune_trials"])
    autotune_warmup = int(config["autotune_warmup"])
    autotune_repeats = int(config["autotune_repeats"])

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
        trials = generate_trials(
            n_qubits=n_qubits,
            depth=depth,
            n_trials=n_trials,
            seed=config_seed,
        )
        if idx == 0:
            first_trials = trials
            first_config = (n_qubits, depth)

        qiskit_batch_size = base_batch_size
        if use_qiskit and autotune_batch:
            tune_tau = softplus_taus[0] if filter_type == "softplus" else DEFAULT_SOFTPLUS_TAU
            qiskit_batch_size = _autotune_qiskit_batch_size(
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
            )
            if progress:
                console.print(f"Auto-tuned Qiskit batch size: {qiskit_batch_size}")

        if idx == 0:
            first_batch_size = qiskit_batch_size

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
            use_qiskit=use_qiskit,
            trials=trials,
            qiskit_batch_size=qiskit_batch_size,
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
            use_qiskit=use_qiskit,
            trials=first_trials,
            qiskit_batch_size=first_batch_size or base_batch_size,
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
