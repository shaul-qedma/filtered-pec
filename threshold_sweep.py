"""
Parameter sweep for threshold/softplus PEC hyperparameters.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from typing import Callable, List, Optional, Tuple

import numpy as np
from rich import box
from rich.console import Console
from rich.table import Table

from exp_window_pec import DEFAULT_QISKIT_BATCH_SIZE
from threshold_pec import benchmark, generate_trials, TrialData

console = Console()
DEFAULT_N_QUBITS = 4
DEFAULT_DEPTH = 4
DEFAULT_QUBITS_RANGE = "4-8"
DEFAULT_DEPTHS_RANGE = "6-14:2"
DEFAULT_MIN_VOLUME = 24
DEFAULT_MAX_VOLUME = 64
DEFAULT_RATIOS = "0.75,1.0,1.5,2.0"
DEFAULT_CONFIGS_PER_RATIO = 2
DEFAULT_MAX_CONFIGS = 0
DEFAULT_N_SAMPLES = 500
DEFAULT_N_TRIALS = 12
DEFAULT_SEED = 42
DEFAULT_W0_DENSITIES = "0.30,0.40,0.50,0.60"
DEFAULT_BETA_PROPS = "0.10,0.15,0.20"
DEFAULT_BETA_THRESHES = "0.10,0.15,0.20,0.25"
DEFAULT_SOFTPLUS_TAUS = "0.5,1.0,2.0"
DEFAULT_SOFTPLUS_TAU = 1.0
DEFAULT_CONFIG_SEED_OFFSET = 1000
RANGE_STEP_DEFAULT = 1
DEFAULT_PROFILE_OUT = "threshold_sweep.prof"
DEFAULT_PROFILE_TOP = 30
DEFAULT_PROFILE_SORT = "cumulative"

def _parse_list(raw: str, cast: Callable) -> List:
    return [cast(x) for x in raw.split(",") if x.strip()]


def _parse_range_list(raw: str) -> List[int]:
    values: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            if ":" in part:
                range_part, step_part = part.split(":", 1)
                step = int(step_part)
            else:
                range_part = part
                step = RANGE_STEP_DEFAULT
            start_str, end_str = range_part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if step < 1:
                raise ValueError("Range step must be positive.")
            values.extend(list(range(start, end + 1, step)))
        else:
            values.append(int(part))
    return sorted(set(values))


def _parse_configs(raw: str) -> List[Tuple[int, int]]:
    configs: List[Tuple[int, int]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Bad config '{part}', expected NxD like 3x2.")
        n_str, d_str = part.split("x", 1)
        configs.append((int(n_str), int(d_str)))
    return configs


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
    errors = np.array([d["error"] for d in results], dtype=float)
    qp_norms = np.array([d["qp_norm"] for d in results], dtype=float)
    ess_vals = [float(val) for val in (d.get("ess") for d in results) if val is not None]
    ess = float(np.mean(ess_vals)) if ess_vals else float("nan")
    above_vals = [float(val) for val in (d.get("above") for d in results) if val is not None]
    above = float(np.mean(above_vals)) if above_vals else float("nan")
    weight_vals = [float(val) for val in (d.get("weight_mean") for d in results) if val is not None]
    weight_mean = float(np.mean(weight_vals)) if weight_vals else float("nan")
    return {
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "qp_norm": float(np.mean(qp_norms)),
        "ess": ess,
        "above": above,
        "weight_mean": weight_mean,
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

    if progress:
        console.print("  Full PEC trials")
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
        progress=progress,
    )
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
                    if progress:
                        label = (
                            f"  {filter_type} ρ={rho:.2f} β_prop={beta_prop:.2f} β_thresh={beta_thresh:.2f}"
                        )
                        if filter_type == "softplus":
                            label = f"{label} τ={tau:.2f}"
                        console.print(label)
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
                        progress=progress,
                    )

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

    if progress:
        console.print("  Threshold trials")
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
        if progress:
            console.print(f"  Softplus τ={tau:.2f} trials")
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


def _run(args: argparse.Namespace) -> None:
    w0_densities = _parse_list(args.threshold_densities, float)
    ratios = _parse_list(args.ratios, float)
    if not w0_densities:
        raise ValueError("At least one threshold density is required.")

    if args.single:
        configs = [(args.n_qubits, args.depth)]
    elif args.configs.strip():
        configs = _parse_configs(args.configs)
    else:
        qubits = _parse_range_list(args.qubits)
        depths = _parse_range_list(args.depths)
        configs = auto_configs(
            qubits=qubits,
            depths=depths,
            min_volume=args.min_volume,
            max_volume=args.max_volume if args.max_volume > 0 else None,
            ratios=ratios,
            per_ratio=max(1, args.configs_per_ratio),
            max_configs=max(0, args.max_configs),
        )
    beta_prop_values = _parse_list(args.beta_props, float)
    beta_thresh_values = _parse_list(args.beta_threshes, float)
    softplus_taus = _parse_list(args.softplus_taus, float)
    benchmark_count = _benchmark_count(
        w0_densities=w0_densities,
        beta_prop_values=beta_prop_values,
        beta_thresh_values=beta_thresh_values,
        filter_type=args.filter_type,
        softplus_taus=softplus_taus,
    )

    timing_rows = []
    t0 = time.perf_counter()
    if not configs:
        raise ValueError("No configurations selected.")
    first_trials: Optional[List[TrialData]] = None
    first_config: Optional[Tuple[int, int]] = None
    for idx, (n_qubits, depth) in enumerate(configs):
        if idx:
            console.rule()
        trials = generate_trials(
            n_qubits=n_qubits,
            depth=depth,
            n_trials=args.n_trials,
            seed=args.seed + idx * DEFAULT_CONFIG_SEED_OFFSET,
        )
        if idx == 0:
            first_trials = trials
            first_config = (n_qubits, depth)
        config_start = time.perf_counter()
        parameter_sweep(
            n_qubits=n_qubits,
            depth=depth,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            seed=args.seed + idx * DEFAULT_CONFIG_SEED_OFFSET,
            w0_densities=w0_densities,
            beta_prop_values=beta_prop_values,
            beta_thresh_values=beta_thresh_values,
            filter_type=args.filter_type,
            softplus_taus=softplus_taus,
            use_qiskit=not args.no_qiskit,
            trials=trials,
            qiskit_batch_size=args.qiskit_batch_size,
            progress=not args.no_progress,
        )
        elapsed = time.perf_counter() - config_start
        if args.timings or args.profile:
            volume = estimate_error_locs(n_qubits, depth)
            timing_rows.append(
                {
                    "config": f"{n_qubits}q_d{depth}",
                    "volume": volume,
                    "benchmarks": benchmark_count,
                    "total_s": elapsed,
                }
            )

    if not args.skip_compare:
        if first_config is None or first_trials is None:
            raise ValueError("No configurations selected for comparison.")
        compare_n, compare_d = first_config
        compare_filters(
            n_qubits=compare_n,
            depth=compare_d,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            seed=args.seed + 1,
            w0_density=w0_densities[0],
            beta_prop=beta_prop_values[0],
            beta_thresh=beta_thresh_values[0],
            softplus_taus=softplus_taus,
            use_qiskit=not args.no_qiskit,
            trials=first_trials,
            qiskit_batch_size=args.qiskit_batch_size,
            progress=not args.no_progress,
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
            per_trial = per_bench / args.n_trials
            per_sample = per_trial / args.n_samples
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


def _run_with_profile(args: argparse.Namespace) -> None:
    profiler = cProfile.Profile()
    profiler.enable()
    _run(args)
    profiler.disable()

    profiler.dump_stats(args.profile_out)
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats(args.profile_sort)
    stats.print_stats(args.profile_top)
    console.rule("Profile Summary")
    console.print(stream.getvalue())
    console.print(f"Profile saved to {args.profile_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep threshold/softplus PEC hyperparameters.")
    parser.add_argument("--n-qubits", type=int, default=DEFAULT_N_QUBITS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated list like 3x2,4x3. Ignored if --single is set.",
    )
    parser.add_argument("--qubits", type=str, default=DEFAULT_QUBITS_RANGE,
                        help="Candidate qubit counts, supports ranges like 3-7 or lists.")
    parser.add_argument("--depths", type=str, default=DEFAULT_DEPTHS_RANGE,
                        help="Candidate depths, supports ranges like 4-12:2 or lists.")
    parser.add_argument("--min-volume", type=int, default=DEFAULT_MIN_VOLUME,
                        help="Minimum error-site volume to include.")
    parser.add_argument("--max-volume", type=int, default=DEFAULT_MAX_VOLUME,
                        help="Maximum error-site volume to include.")
    parser.add_argument("--ratios", type=str, default=DEFAULT_RATIOS,
                        help="Target depth/width ratios to explore.")
    parser.add_argument("--configs-per-ratio", type=int, default=DEFAULT_CONFIGS_PER_RATIO,
                        help="Configs to pick per ratio target.")
    parser.add_argument("--max-configs", type=int, default=DEFAULT_MAX_CONFIGS,
                        help="Cap total configs (0 for no cap).")
    parser.add_argument("--single", action="store_true", help="Use --n-qubits/--depth only.")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--threshold-densities",
        type=str,
        default=DEFAULT_W0_DENSITIES,
        help="Comma-separated densities (w0 = ceil(rho * n_locs)).",
    )
    parser.add_argument("--beta-props", type=str, default=DEFAULT_BETA_PROPS)
    parser.add_argument("--beta-threshes", type=str, default=DEFAULT_BETA_THRESHES)
    parser.add_argument("--filter-type", choices=["threshold", "softplus"], default="threshold")
    parser.add_argument("--softplus-taus", type=str, default=DEFAULT_SOFTPLUS_TAUS)
    parser.add_argument("--no-qiskit", action="store_true", help="Disable Qiskit simulation")
    parser.add_argument("--no-progress", action="store_true", help="Disable trial progress logging")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--timings", action="store_true", help="Print per-config timing summary")
    parser.add_argument("--profile", action="store_true", help="Run cProfile and print a summary")
    parser.add_argument("--qiskit-batch-size", type=int, default=DEFAULT_QISKIT_BATCH_SIZE)
    parser.add_argument("--profile-out", type=str, default=DEFAULT_PROFILE_OUT)
    parser.add_argument("--profile-top", type=int, default=DEFAULT_PROFILE_TOP)
    parser.add_argument(
        "--profile-sort",
        choices=["cumulative", "tottime"],
        default=DEFAULT_PROFILE_SORT,
    )
    args = parser.parse_args()

    if args.profile:
        _run_with_profile(args)
    else:
        _run(args)


if __name__ == "__main__":
    main()
