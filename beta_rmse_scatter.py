"""
RMSE vs beta scatter plot with confidence intervals for multiple configs.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from exp_window_pec import pec_estimate_qiskit
from pec_shared import (
    STANDARD_GATES,
    NoisySimulator,
    error_locations,
    random_circuit,
    random_noise_model,
    random_observable,
    random_product_state,
)

GATE_P_I_RANGES = {
    "CNOT": (0.80, 0.90),
    "CNOT_R": (0.80, 0.90),
    "CZ": (0.88, 0.96),
    "SWAP": (0.78, 0.88),
    "default": (0.85, 0.95),
}


@dataclass
class BetaRMSE:
    beta: float
    rmse: float
    ci_low: float
    ci_high: float


@dataclass
class ConfigRMSE:
    n_qubits: int
    depth: int
    stats: List[BetaRMSE]


def parse_betas(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def parse_configs(raw: str) -> List[Tuple[int, int]]:
    configs = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" not in part:
            raise ValueError(f"Bad config '{part}', expected NxD like 3x2.")
        n_str, d_str = part.split("x", 1)
        configs.append((int(n_str), int(d_str)))
    return configs


def bootstrap_rmse(
    errors: List[float],
    confidence: float,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:
    errors_arr = np.asarray(errors, dtype=float)
    rmse = float(np.sqrt(np.mean(errors_arr ** 2))) if len(errors_arr) else 0.0
    if len(errors_arr) < 2 or n_boot <= 0:
        return rmse, rmse, rmse
    idx = rng.integers(0, len(errors_arr), size=(n_boot, len(errors_arr)))
    sampled = errors_arr[idx]
    rmses = np.sqrt(np.mean(sampled ** 2, axis=1))
    alpha = (1.0 - confidence) / 2.0 * 100.0
    ci_low = float(np.percentile(rmses, alpha))
    ci_high = float(np.percentile(rmses, 100.0 - alpha))
    return rmse, ci_low, ci_high


def run_trials_for_config(
    n_qubits: int,
    depth: int,
    betas: List[float],
    n_samples: int,
    n_trials: int,
    seed: int,
    confidence: float,
    n_boot: int,
    p_I_range,
) -> ConfigRMSE:
    errors: Dict[float, List[float]] = {b: [] for b in betas}

    for t in range(n_trials):
        rng = np.random.default_rng(seed + 1000 * t + 31 * n_qubits + depth)
        circuit = random_circuit(n_qubits, depth, rng)
        noise = random_noise_model(rng, p_I_range=p_I_range)
        init = random_product_state(n_qubits, rng)
        obs = random_observable(n_qubits, rng)
        locs = error_locations(circuit, noise)

        sim = NoisySimulator(STANDARD_GATES, noise)
        ideal = sim.ideal(circuit, obs, init)

        for i, beta in enumerate(betas):
            est = pec_estimate_qiskit(
                circuit,
                obs,
                init,
                noise,
                locs,
                beta=beta,
                n_samples=n_samples,
                seed=seed + 50000 * t + i,
            )
            errors[beta].append(float(est.mean - ideal))

    stats = []
    rng_boot = np.random.default_rng(seed + 99991 + n_qubits * 11 + depth)
    for beta in betas:
        rmse, ci_low, ci_high = bootstrap_rmse(
            errors[beta], confidence, n_boot, rng_boot
        )
        stats.append(BetaRMSE(beta=beta, rmse=rmse, ci_low=ci_low, ci_high=ci_high))

    return ConfigRMSE(n_qubits=n_qubits, depth=depth, stats=stats)


def _ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if vmax == vmin:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + i * step for i in range(count)]


def write_svg(
    results: List[ConfigRMSE],
    out_path: str,
    title: str,
) -> None:
    width, height = 960, 560
    margin = {"left": 90, "right": 30, "top": 50, "bottom": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    betas = sorted({s.beta for r in results for s in r.stats})
    x_min, x_max = min(betas), max(betas)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    x_pad = 0.1 * (x_max - x_min)
    x_min -= x_pad
    x_max += x_pad

    y_min = min(s.ci_low for r in results for s in r.stats)
    y_max = max(s.ci_high for r in results for s in r.stats)
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1
    y_pad = 0.12 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def x_px(x: float) -> float:
        return margin["left"] + (x - x_min) / (x_max - x_min) * plot_w

    def y_px(y: float) -> float:
        return margin["top"] + (y_max - y) / (y_max - y_min) * plot_h

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
        f'font-family="Arial" font-size="18" fill="#222">{title}</text>'
    )

    x0 = margin["left"]
    y0 = margin["top"] + plot_h
    x1 = margin["left"] + plot_w
    y1 = margin["top"]
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" />')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" />')

    for xv in _ticks(x_min, x_max):
        xp = x_px(xv)
        parts.append(f'<line x1="{xp:.1f}" y1="{y0}" x2="{xp:.1f}" y2="{y0+6}" stroke="#222"/>')
        parts.append(
            f'<text x="{xp:.1f}" y="{y0+22}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#222">{xv:.2f}</text>'
        )
    for yv in _ticks(y_min, y_max):
        yp = y_px(yv)
        parts.append(f'<line x1="{x0-6}" y1="{yp:.1f}" x2="{x0}" y2="{yp:.1f}" stroke="#222"/>')
        parts.append(
            f'<text x="{x0-10}" y="{yp+4:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#222">{yv:.3f}</text>'
        )

    parts.append(
        f'<text x="{x0 + plot_w/2:.1f}" y="{height-25}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222">beta</text>'
    )
    parts.append(
        f'<text x="22" y="{margin["top"] + plot_h/2:.1f}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222" '
        f'transform="rotate(-90 22 {margin["top"] + plot_h/2:.1f})">RMSE</text>'
    )

    n_cfg = len(results)
    x_range = x_max - x_min
    band_w = 0.035 * x_range
    jitter_step = band_w * 0.7

    for idx, cfg in enumerate(results):
        color = colors[idx % len(colors)]
        jitter = (idx - (n_cfg - 1) / 2.0) * jitter_step
        for s in cfg.stats:
            x_center = s.beta + jitter
            xp = x_px(x_center)
            y_center = y_px(s.rmse)
            x0b = x_px(x_center - band_w / 2.0)
            x1b = x_px(x_center + band_w / 2.0)
            y0b = y_px(s.ci_high)
            y1b = y_px(s.ci_low)
            w = max(1.0, x1b - x0b)
            h = max(1.0, y1b - y0b)
            parts.append(
                f'<rect x="{x0b:.1f}" y="{y0b:.1f}" width="{w:.1f}" height="{h:.1f}" '
                f'fill="{color}" opacity="0.12" stroke="none" />'
            )
            parts.append(f'<circle cx="{xp:.1f}" cy="{y_center:.1f}" r="5" fill="{color}" />')

    # Legend
    legend_x = width - margin["right"] - 160
    legend_y = margin["top"] + 10
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y - 16}" width="160" height="{18 * n_cfg + 10}" '
        f'fill="#f7f7f7" stroke="#ddd"/>'
    )
    for idx, cfg in enumerate(results):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 18
        label = f"{cfg.n_qubits}q d{cfg.depth}"
        parts.append(f'<rect x="{legend_x + 8}" y="{y - 9}" width="10" height="10" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 24}" y="{y}" font-family="Arial" font-size="12" fill="#222">'
            f'{label}</text>'
        )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="RMSE vs beta for multiple configs.")
    parser.add_argument("--betas", default="0.00,0.05,0.10,0.20", help="Comma-separated beta values.")
    parser.add_argument(
        "--configs",
        default="2x2,3x3,3x4,4x3,4x4,5x3",
        help="Comma-separated configs like 3x2.",
    )
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument(
        "--noise-profile",
        choices=["uniform", "gate"],
        default="gate",
        help="Noise profile: uniform p_I range or gate-dependent ranges.",
    )
    parser.add_argument("--out", default="beta_rmse_scatter.svg")
    args = parser.parse_args()

    betas = parse_betas(args.betas)
    configs = parse_configs(args.configs)

    t0 = time.time()
    p_I_range = (0.85, 0.95) if args.noise_profile == "uniform" else GATE_P_I_RANGES

    results = []
    for n_qubits, depth in configs:
        results.append(
            run_trials_for_config(
                n_qubits=n_qubits,
                depth=depth,
                betas=betas,
                n_samples=args.n_samples,
                n_trials=args.n_trials,
                seed=args.seed,
                confidence=args.confidence,
                n_boot=args.bootstrap,
                p_I_range=p_I_range,
            )
        )

    print("config, beta, rmse, ci_low, ci_high")
    for cfg in results:
        label = f"{cfg.n_qubits}q_d{cfg.depth}"
        for s in cfg.stats:
            print(f"{label}, {s.beta:.4f}, {s.rmse:.6f}, {s.ci_low:.6f}, {s.ci_high:.6f}")

    title = (
        f"RMSE vs beta (n_samples={args.n_samples}, n_trials={args.n_trials}, "
        f"noise={args.noise_profile})"
    )
    write_svg(results, args.out, title)
    print(f"Wrote {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
