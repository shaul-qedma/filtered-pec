"""
Compare exponential-window PEC beta values using Qiskit simulation.

Outputs an SVG scatter plot of bias vs effective samples with confidence
interval shading for each beta.
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Dict, List

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


@dataclass
class BetaStats:
    beta: float
    mean_bias: float
    bias_ci: float
    mean_eff_samples: float
    eff_ci: float


def parse_betas(raw: str) -> List[float]:
    return [float(x) for x in raw.split(",") if x.strip()]


def z_value(confidence: float) -> float:
    conf = round(confidence, 2)
    mapping = {
        0.68: 1.0,
        0.90: 1.645,
        0.95: 1.96,
        0.99: 2.576,
    }
    return mapping.get(conf, 1.96)


def ci_half_width(values: List[float], confidence: float) -> float:
    if len(values) < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    return z_value(confidence) * std / math.sqrt(len(values))


def run_trials(
    betas: List[float],
    n_qubits: int,
    depth: int,
    n_samples: int,
    n_trials: int,
    seed: int,
    confidence: float,
) -> List[BetaStats]:
    biases: Dict[float, List[float]] = {b: [] for b in betas}
    eff_samples: Dict[float, List[float]] = {b: [] for b in betas}

    for t in range(n_trials):
        rng = np.random.default_rng(seed + t)
        circuit = random_circuit(n_qubits, depth, rng)
        noise = random_noise_model(rng)
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
                seed=seed + 1000 * t + i,
            )
            bias = est.mean - ideal
            eff = n_samples / (est.qp_norm ** 2)
            biases[beta].append(float(bias))
            eff_samples[beta].append(float(eff))

    stats = []
    for beta in betas:
        b_vals = biases[beta]
        e_vals = eff_samples[beta]
        stats.append(
            BetaStats(
                beta=beta,
                mean_bias=float(np.mean(b_vals)),
                bias_ci=ci_half_width(b_vals, confidence),
                mean_eff_samples=float(np.mean(e_vals)),
                eff_ci=ci_half_width(e_vals, confidence),
            )
        )
    return stats


def _ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if vmax == vmin:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + i * step for i in range(count)]


def write_svg_scatter(stats: List[BetaStats], out_path: str, title: str) -> None:
    width, height = 900, 560
    margin = {"left": 90, "right": 30, "top": 50, "bottom": 80}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    x_vals = [s.mean_eff_samples for s in stats]
    y_vals = [s.mean_bias for s in stats]
    x_errs = [s.eff_ci for s in stats]
    y_errs = [s.bias_ci for s in stats]

    x_min = min(x - dx for x, dx in zip(x_vals, x_errs))
    x_max = max(x + dx for x, dx in zip(x_vals, x_errs))
    y_min = min(y - dy for y, dy in zip(y_vals, y_errs))
    y_max = max(y + dy for y, dy in zip(y_vals, y_errs))

    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1

    x_pad = 0.08 * (x_max - x_min)
    y_pad = 0.12 * (y_max - y_min)
    x_min -= x_pad
    x_max += x_pad
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

    # Axes
    x0 = margin["left"]
    y0 = margin["top"] + plot_h
    x1 = margin["left"] + plot_w
    y1 = margin["top"]
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#222" />')
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#222" />')

    # Ticks and labels
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

    # Axis labels
    parts.append(
        f'<text x="{x0 + plot_w/2:.1f}" y="{height-25}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222">Effective samples (n_samples / qp_norm^2)</text>'
    )
    parts.append(
        f'<text x="22" y="{margin["top"] + plot_h/2:.1f}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222" transform="rotate(-90 22 {margin["top"] + plot_h/2:.1f})">'
        f'Bias (estimate - ideal)</text>'
    )

    # Points with CI boxes
    for idx, s in enumerate(stats):
        color = colors[idx % len(colors)]
        xp = x_px(s.mean_eff_samples)
        yp = y_px(s.mean_bias)
        x0b = x_px(s.mean_eff_samples - s.eff_ci)
        x1b = x_px(s.mean_eff_samples + s.eff_ci)
        y0b = y_px(s.mean_bias + s.bias_ci)
        y1b = y_px(s.mean_bias - s.bias_ci)
        w = max(1.0, x1b - x0b)
        h = max(1.0, y1b - y0b)
        parts.append(
            f'<rect x="{x0b:.1f}" y="{y0b:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{color}" opacity="0.12" stroke="none" />'
        )
        parts.append(f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="5" fill="{color}" />')
        parts.append(
            f'<text x="{xp + 8:.1f}" y="{yp - 8:.1f}" font-family="Arial" '
            f'font-size="12" fill="#222">beta={s.beta:.2f}</text>'
        )

    parts.append("</svg>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare beta values with Qiskit PEC.")
    parser.add_argument("--betas", default="0.00,0.05,0.10,0.20", help="Comma-separated beta values.")
    parser.add_argument("--n-qubits", type=int, default=3)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n-samples", type=int, default=40)
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--out", default="beta_scatter.svg")
    args = parser.parse_args()

    betas = parse_betas(args.betas)
    t0 = time.time()
    stats = run_trials(
        betas=betas,
        n_qubits=args.n_qubits,
        depth=args.depth,
        n_samples=args.n_samples,
        n_trials=args.n_trials,
        seed=args.seed,
        confidence=args.confidence,
    )

    print("beta, mean_bias, bias_ci, mean_eff_samples, eff_ci")
    for s in stats:
        print(f"{s.beta:.4f}, {s.mean_bias:.6f}, {s.bias_ci:.6f}, {s.mean_eff_samples:.2f}, {s.eff_ci:.2f}")

    title = f"Bias vs Effective Samples (n_trials={args.n_trials}, n_samples={args.n_samples})"
    write_svg_scatter(stats, args.out, title)
    print(f"Wrote {args.out} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
