"""
Parameter sweep for threshold/softplus PEC hyperparameters.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, List, Optional, Tuple

import numpy as np

from threshold_pec import benchmark


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
                step = 1
            start_str, end_str = range_part.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if step <= 0:
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
    ess_vals = [d.get("ess") for d in results if "ess" in d]
    ess = float(np.mean(ess_vals)) if ess_vals else float("nan")
    above_vals = [d.get("above") for d in results if "above" in d]
    above = float(np.mean(above_vals)) if above_vals else float("nan")
    weight_vals = [d.get("weight_mean") for d in results if "weight_mean" in d]
    weight_mean = float(np.mean(weight_vals)) if weight_vals else float("nan")
    return {
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "qp_norm": float(np.mean(qp_norms)),
        "ess": ess,
        "above": above,
        "weight_mean": weight_mean,
    }


def parameter_sweep(
    n_qubits: int,
    depth: int,
    n_samples: int,
    n_trials: int,
    seed: int,
    w0_values: List[int],
    w0_densities: List[float],
    use_density: bool,
    w0_mode: str,
    beta_prop_values: List[float],
    beta_thresh_values: List[float],
    filter_type: str,
    softplus_taus: List[float],
    use_qiskit: bool,
) -> List[dict]:
    print("=" * 70)
    print("THRESHOLD PEC PARAMETER SWEEP")
    print("=" * 70)

    volume = estimate_error_locs(n_qubits, depth)
    print(f"\nCircuit: {n_qubits} qubits, depth {depth}, volume={volume}, w0_mode={w0_mode}")
    print(f"Samples: {n_samples}, Trials: {n_trials}, Qiskit={use_qiskit}")

    data_full = benchmark(
        n_qubits=n_qubits,
        depth=depth,
        w0=0,
        beta_prop=0.0,
        beta_thresh=0.0,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=seed,
        filter_type="threshold",
        softplus_tau=1.0,
        use_qiskit=use_qiskit,
    )
    full_errors = np.array([d["error"] for d in data_full["results"]["full_pec"]])
    full_rmse = float(np.sqrt(np.mean(full_errors**2)))
    full_qp_norm = float(np.mean([d["qp_norm"] for d in data_full["results"]["full_pec"]]))
    full_bias = float(np.mean(full_errors))

    results = []

    if use_density:
        if filter_type == "softplus":
            print(
                f"\n{'ρ':>4} {'w0':>6} {'β_prop':>8} {'β_thresh':>10} {'τ':>6} "
                f"{'qp_norm':>8} {'Bias':>10} {'RMSE':>10} {'ESS':>8} {'w_mean':>8} {'p_gt_w0':>8} {'vs Full':>10}"
            )
            print("-" * 112)
        else:
            print(
                f"\n{'ρ':>4} {'w0':>6} {'β_prop':>8} {'β_thresh':>10} "
                f"{'qp_norm':>8} {'Bias':>10} {'RMSE':>10} {'ESS':>8} {'w_mean':>8} {'p_gt_w0':>8} {'vs Full':>10}"
            )
            print("-" * 104)
    else:
        if filter_type == "softplus":
            print(
                f"\n{'w0':>4} {'β_prop':>8} {'β_thresh':>10} {'τ':>6} "
                f"{'qp_norm':>8} {'Bias':>10} {'RMSE':>10} {'ESS':>8} {'w_mean':>8} {'p_gt_w0':>8} {'vs Full':>10}"
            )
            print("-" * 106)
        else:
            print(
                f"\n{'w0':>4} {'β_prop':>8} {'β_thresh':>10} "
                f"{'qp_norm':>8} {'Bias':>10} {'RMSE':>10} {'ESS':>8} {'w_mean':>8} {'p_gt_w0':>8} {'vs Full':>10}"
            )
            print("-" * 98)

    if filter_type == "softplus":
        if use_density:
            print(
                f"{'Full':>4} {'-':>6} {'-':>8} {'-':>10} {'-':>6} "
                f"{full_qp_norm:>8.2f} {full_bias:>10.4f} {full_rmse:>10.4f} {'-':>8} {'-':>8} {'-':>8} {'baseline':>10}"
            )
        else:
            print(
                f"{'Full':>4} {'-':>8} {'-':>10} {'-':>6} "
                f"{full_qp_norm:>8.2f} {full_bias:>10.4f} {full_rmse:>10.4f} {'-':>8} {'-':>8} {'-':>8} {'baseline':>10}"
            )
    else:
        if use_density:
            print(
                f"{'Full':>4} {'-':>6} {'-':>8} {'-':>10} "
                f"{full_qp_norm:>8.2f} {full_bias:>10.4f} {full_rmse:>10.4f} {'-':>8} {'-':>8} {'-':>8} {'baseline':>10}"
            )
        else:
            print(
                f"{'Full':>4} {'-':>8} {'-':>10} "
                f"{full_qp_norm:>8.2f} {full_bias:>10.4f} {full_rmse:>10.4f} {'-':>8} {'-':>8} {'-':>8} {'baseline':>10}"
            )

    sweep_w0 = w0_values if not use_density else [0]
    sweep_rho = w0_densities if use_density else [None]

    for w0 in sweep_w0:
        for rho in sweep_rho:
            if use_density and rho is None:
                continue
            for beta_prop in beta_prop_values:
                for beta_thresh in beta_thresh_values:
                    taus = softplus_taus if filter_type == "softplus" else [1.0]
                    for tau in taus:
                        data = benchmark(
                            n_qubits=n_qubits,
                            depth=depth,
                            w0=w0,
                            beta_prop=beta_prop,
                            beta_thresh=beta_thresh,
                            n_samples=n_samples,
                            n_trials=n_trials,
                            seed=seed,
                            filter_type=filter_type,
                            softplus_tau=tau,
                            use_qiskit=use_qiskit,
                            w0_density=rho if use_density else None,
                            w0_mode=w0_mode,
                        )

                        stats = _summary_from_filtered(data["results"]["filtered"])
                        vs_full = 100 * (stats["rmse"] / full_rmse - 1)

                        row = {
                            "w0": w0,
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
                            "w0_min": data["config"]["w0_min"],
                            "w0_max": data["config"]["w0_max"],
                        }
                        results.append(row)

                        if filter_type == "softplus":
                            if use_density:
                                print(
                                    f"{rho:>4.2f} {data['config']['w0_mean']:>6.1f} {beta_prop:>8.2f} "
                                    f"{beta_thresh:>10.2f} {tau:>6.2f} {stats['qp_norm']:>8.2f} "
                                    f"{stats['bias']:>10.4f} {stats['rmse']:>10.4f} "
                                    f"{stats['ess']:>8.1f} {stats['weight_mean']:>8.2f} "
                                    f"{stats['above']:>8.2f} {vs_full:>+10.1f}%"
                                )
                            else:
                                print(
                                    f"{w0:>4} {beta_prop:>8.2f} {beta_thresh:>10.2f} {tau:>6.2f} "
                                    f"{stats['qp_norm']:>8.2f} {stats['bias']:>10.4f} {stats['rmse']:>10.4f} "
                                    f"{stats['ess']:>8.1f} {stats['weight_mean']:>8.2f} "
                                    f"{stats['above']:>8.2f} {vs_full:>+10.1f}%"
                                )
                        else:
                            if use_density:
                                print(
                                    f"{rho:>4.2f} {data['config']['w0_mean']:>6.1f} {beta_prop:>8.2f} "
                                    f"{beta_thresh:>10.2f} {stats['qp_norm']:>8.2f} {stats['bias']:>10.4f} "
                                    f"{stats['rmse']:>10.4f} {stats['ess']:>8.1f} {stats['weight_mean']:>8.2f} "
                                    f"{stats['above']:>8.2f} {vs_full:>+10.1f}%"
                                )
                            else:
                                print(
                                    f"{w0:>4} {beta_prop:>8.2f} {beta_thresh:>10.2f} "
                                    f"{stats['qp_norm']:>8.2f} {stats['bias']:>10.4f} {stats['rmse']:>10.4f} "
                                    f"{stats['ess']:>8.1f} {stats['weight_mean']:>8.2f} "
                                    f"{stats['above']:>8.2f} {vs_full:>+10.1f}%"
                                )

    best = min(results, key=lambda r: r["rmse"])
    print("\nBest configuration:")
    if filter_type == "softplus":
        if use_density:
            print(
                f"  ρ={best['w0_density']}, w0≈{best['w0_mean']:.1f}, "
                f"β_prop={best['beta_prop']}, β_thresh={best['beta_thresh']}, τ={best['tau']}"
            )
        else:
            print(
                f"  w0={best['w0']}, β_prop={best['beta_prop']}, β_thresh={best['beta_thresh']}, τ={best['tau']}"
            )
    else:
        if use_density:
            print(
                f"  ρ={best['w0_density']}, w0≈{best['w0_mean']:.1f}, "
                f"β_prop={best['beta_prop']}, β_thresh={best['beta_thresh']}"
            )
        else:
            print(f"  w0={best['w0']}, β_prop={best['beta_prop']}, β_thresh={best['beta_thresh']}")
    print(f"  RMSE = {best['rmse']:.4f} ({best['vs_full']:+.1f}% vs Full PEC)")

    return results


def compare_filters(
    n_qubits: int,
    depth: int,
    n_samples: int,
    n_trials: int,
    seed: int,
    w0: int,
    w0_density: Optional[float],
    w0_mode: str,
    beta_prop: float,
    beta_thresh: float,
    softplus_taus: List[float],
    use_qiskit: bool,
) -> None:
    print("\n" + "=" * 70)
    print("THRESHOLD vs SOFTPLUS FILTER COMPARISON")
    print("=" * 70)

    data_thresh = benchmark(
        n_qubits=n_qubits,
        depth=depth,
        w0=w0,
        beta_prop=beta_prop,
        beta_thresh=beta_thresh,
        n_samples=n_samples,
        n_trials=n_trials,
        seed=seed,
        filter_type="threshold",
        softplus_tau=1.0,
        use_qiskit=use_qiskit,
        w0_density=w0_density,
        w0_mode=w0_mode,
    )
    thresh_stats = _summary_from_filtered(data_thresh["results"]["filtered"])

    if w0_density is not None:
        w0_label = f"ρ={w0_density}, w0≈{data_thresh['config']['w0_mean']:.1f}"
    else:
        w0_label = f"w0={w0}"

    print(f"\nConfig: {n_qubits}q, depth={depth}, {w0_label}, β_prop={beta_prop}, β_thresh={beta_thresh}")
    print(f"{'Filter':<20} {'Bias':>10} {'RMSE':>10} {'w_mean':>8} {'p_gt_w0':>8}")
    print("-" * 63)
    print(
        f"{'Threshold':<20} {thresh_stats['bias']:>10.4f} {thresh_stats['rmse']:>10.4f} "
        f"{thresh_stats['weight_mean']:>8.2f} {thresh_stats['above']:>8.2f}"
    )

    for tau in softplus_taus:
        data_soft = benchmark(
            n_qubits=n_qubits,
            depth=depth,
            w0=w0,
            beta_prop=beta_prop,
            beta_thresh=beta_thresh,
            n_samples=n_samples,
            n_trials=n_trials,
            seed=seed,
            filter_type="softplus",
            softplus_tau=tau,
            use_qiskit=use_qiskit,
            w0_density=w0_density,
            w0_mode=w0_mode,
        )
        soft_stats = _summary_from_filtered(data_soft["results"]["filtered"])
        print(
            f"{'Softplus τ=' + str(tau):<20} {soft_stats['bias']:>10.4f} {soft_stats['rmse']:>10.4f} "
            f"{soft_stats['weight_mean']:>8.2f} {soft_stats['above']:>8.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep threshold/softplus PEC hyperparameters.")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument(
        "--configs",
        type=str,
        default="",
        help="Comma-separated list like 3x2,4x3. Ignored if --single is set.",
    )
    parser.add_argument("--qubits", type=str, default="4-8",
                        help="Candidate qubit counts, supports ranges like 3-7 or lists.")
    parser.add_argument("--depths", type=str, default="6-14:2",
                        help="Candidate depths, supports ranges like 4-12:2 or lists.")
    parser.add_argument("--min-volume", type=int, default=24,
                        help="Minimum error-site volume to include.")
    parser.add_argument("--max-volume", type=int, default=64,
                        help="Maximum error-site volume to include.")
    parser.add_argument("--ratios", type=str, default="0.75,1.0,1.5,2.0",
                        help="Target depth/width ratios to explore.")
    parser.add_argument("--configs-per-ratio", type=int, default=2,
                        help="Configs to pick per ratio target.")
    parser.add_argument("--max-configs", type=int, default=0,
                        help="Cap total configs (0 for no cap).")
    parser.add_argument("--single", action="store_true", help="Use --n-qubits/--depth only.")
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w0s", type=str, default="")
    parser.add_argument("--w0-densities", type=str, default="0.30,0.40,0.50,0.60")
    parser.add_argument("--w0-mode", choices=["volume", "expected", "tail"], default="volume")
    parser.add_argument("--beta-props", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--beta-threshes", type=str, default="0.10,0.15,0.20,0.25")
    parser.add_argument("--filter-type", choices=["threshold", "softplus"], default="threshold")
    parser.add_argument("--softplus-taus", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--no-qiskit", action="store_true", help="Disable Qiskit simulation")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    w0_values = _parse_list(args.w0s, int) if args.w0s.strip() else []
    w0_densities = _parse_list(args.w0_densities, float)
    ratios = _parse_list(args.ratios, float)

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

    t0 = time.time()
    use_density = not w0_values
    if not configs:
        raise ValueError("No configurations selected.")
    for idx, (n_qubits, depth) in enumerate(configs):
        if idx:
            print("\n" + "=" * 70)
        parameter_sweep(
            n_qubits=n_qubits,
            depth=depth,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            seed=args.seed + idx * 1000,
            w0_values=w0_values,
            w0_densities=w0_densities,
            use_density=use_density,
            w0_mode=args.w0_mode,
            beta_prop_values=beta_prop_values,
            beta_thresh_values=beta_thresh_values,
            filter_type=args.filter_type,
            softplus_taus=softplus_taus,
            use_qiskit=not args.no_qiskit,
        )

    if not args.skip_compare:
        compare_n, compare_d = configs[0]
        compare_filters(
            n_qubits=compare_n,
            depth=compare_d,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            seed=args.seed + 1,
            w0=w0_values[0] if w0_values else 0,
            w0_density=w0_densities[0] if use_density else None,
            w0_mode=args.w0_mode,
            beta_prop=beta_prop_values[0],
            beta_thresh=beta_thresh_values[0],
            softplus_taus=softplus_taus,
            use_qiskit=not args.no_qiskit,
        )

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
