"""
Parameter sweep for threshold/softplus PEC hyperparameters.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, List

import numpy as np

from threshold_pec import benchmark


def _parse_list(raw: str, cast: Callable) -> List:
    return [cast(x) for x in raw.split(",") if x.strip()]


def _summary_from_filtered(results: List[dict]) -> dict:
    errors = np.array([d["error"] for d in results], dtype=float)
    gammas = np.array([d["gamma"] for d in results], dtype=float)
    ess_vals = [d.get("ess") for d in results if "ess" in d]
    ess = float(np.mean(ess_vals)) if ess_vals else float("nan")
    return {
        "bias": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "gamma": float(np.mean(gammas)),
        "ess": ess,
    }


def parameter_sweep(
    n_qubits: int,
    depth: int,
    n_samples: int,
    n_trials: int,
    seed: int,
    w0_values: List[int],
    beta_prop_values: List[float],
    beta_thresh_values: List[float],
    filter_type: str,
    softplus_taus: List[float],
    use_qiskit: bool,
) -> List[dict]:
    print("=" * 70)
    print("THRESHOLD PEC PARAMETER SWEEP")
    print("=" * 70)

    print(f"\nCircuit: {n_qubits} qubits, depth {depth}")
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
    full_gamma = float(np.mean([d["gamma"] for d in data_full["results"]["full_pec"]]))

    results = []

    if filter_type == "softplus":
        print(f"\n{'w0':>4} {'β_prop':>8} {'β_thresh':>10} {'τ':>6} {'γ':>8} {'Bias':>10} {'RMSE':>10} {'ESS':>8} {'vs Full':>10}")
        print("-" * 88)
    else:
        print(f"\n{'w0':>4} {'β_prop':>8} {'β_thresh':>10} {'γ':>8} {'Bias':>10} {'RMSE':>10} {'ESS':>8} {'vs Full':>10}")
        print("-" * 80)

    for w0 in w0_values:
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
                    )

                    stats = _summary_from_filtered(data["results"]["filtered"])
                    vs_full = 100 * (stats["rmse"] / full_rmse - 1)

                    row = {
                        "w0": w0,
                        "beta_prop": beta_prop,
                        "beta_thresh": beta_thresh,
                        "tau": tau,
                        "gamma": stats["gamma"],
                        "bias": stats["bias"],
                        "rmse": stats["rmse"],
                        "ess": stats["ess"],
                        "vs_full": vs_full,
                    }
                    results.append(row)

                    if filter_type == "softplus":
                        print(
                            f"{w0:>4} {beta_prop:>8.2f} {beta_thresh:>10.2f} {tau:>6.2f} "
                            f"{stats['gamma']:>8.2f} {stats['bias']:>10.4f} {stats['rmse']:>10.4f} "
                            f"{stats['ess']:>8.1f} {vs_full:>+10.1f}%"
                        )
                    else:
                        print(
                            f"{w0:>4} {beta_prop:>8.2f} {beta_thresh:>10.2f} "
                            f"{stats['gamma']:>8.2f} {stats['bias']:>10.4f} {stats['rmse']:>10.4f} "
                            f"{stats['ess']:>8.1f} {vs_full:>+10.1f}%"
                        )

    best = min(results, key=lambda r: r["rmse"])
    print("\nBest configuration:")
    if filter_type == "softplus":
        print(
            f"  w0={best['w0']}, β_prop={best['beta_prop']}, β_thresh={best['beta_thresh']}, τ={best['tau']}"
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
    )
    thresh_stats = _summary_from_filtered(data_thresh["results"]["filtered"])

    print(f"\nConfig: {n_qubits}q, depth={depth}, w0={w0}, β_prop={beta_prop}, β_thresh={beta_thresh}")
    print(f"{'Filter':<20} {'Bias':>10} {'RMSE':>10}")
    print("-" * 45)
    print(f"{'Threshold':<20} {thresh_stats['bias']:>10.4f} {thresh_stats['rmse']:>10.4f}")

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
        )
        soft_stats = _summary_from_filtered(data_soft["results"]["filtered"])
        print(f"{'Softplus τ=' + str(tau):<20} {soft_stats['bias']:>10.4f} {soft_stats['rmse']:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep threshold/softplus PEC hyperparameters.")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--n-samples", type=int, default=150)
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w0s", type=str, default="1,2,3")
    parser.add_argument("--beta-props", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--beta-threshes", type=str, default="0.10,0.15,0.20,0.25")
    parser.add_argument("--filter-type", choices=["threshold", "softplus"], default="threshold")
    parser.add_argument("--softplus-taus", type=str, default="0.5,1.0,2.0")
    parser.add_argument("--no-qiskit", action="store_true", help="Disable Qiskit simulation")
    parser.add_argument("--skip-compare", action="store_true")
    args = parser.parse_args()

    w0_values = _parse_list(args.w0s, int)
    beta_prop_values = _parse_list(args.beta_props, float)
    beta_thresh_values = _parse_list(args.beta_threshes, float)
    softplus_taus = _parse_list(args.softplus_taus, float)

    t0 = time.time()
    parameter_sweep(
        n_qubits=args.n_qubits,
        depth=args.depth,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        seed=args.seed,
        w0_values=w0_values,
        beta_prop_values=beta_prop_values,
        beta_thresh_values=beta_thresh_values,
        filter_type=args.filter_type,
        softplus_taus=softplus_taus,
        use_qiskit=not args.no_qiskit,
    )

    if not args.skip_compare:
        compare_filters(
            n_qubits=args.n_qubits,
            depth=args.depth,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            seed=args.seed + 1,
            w0=w0_values[0],
            beta_prop=beta_prop_values[0],
            beta_thresh=beta_thresh_values[0],
            softplus_taus=softplus_taus,
            use_qiskit=not args.no_qiskit,
        )

    print(f"\nTotal time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
