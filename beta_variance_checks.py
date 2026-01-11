"""
Variance checks for exponential-window PEC.

1) Summary across random trials: mean gamma/std per beta.
2) Fixed instance: empirical variance vs beta and n_samples.
"""

import argparse
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np

from exp_window_pec import pec_estimate_qiskit
from beta_rmse_scatter import GATE_P_I_RANGES, parse_betas, parse_configs
from pec_shared import (
    STANDARD_GATES,
    NoisySimulator,
    error_locations,
    random_circuit,
    random_noise_model,
    random_observable,
    random_product_state,
)


def parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


@dataclass
class SummaryPoint:
    beta: float
    mean_gamma: float
    mean_std: float
    mean_bias: float


@dataclass
class SummarySeries:
    n_qubits: int
    depth: int
    points: List[SummaryPoint]


@dataclass
class FixedPoint:
    beta: float
    mean_gamma: float
    mean_bias: float
    emp_std: float
    mean_est_std: float


@dataclass
class FixedSeries:
    n_samples: int
    points: List[FixedPoint]


def _ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if vmax == vmin:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + i * step for i in range(count)]


def write_facets_metric_svg(
    series_list: List[SummarySeries],
    value_fn: Callable[[SummaryPoint], float],
    y_label: str,
    out_path: str,
    title: str,
    cols: int,
) -> None:
    n_cfg = len(series_list)
    cols = max(1, min(cols, n_cfg))
    rows = int(math.ceil(n_cfg / cols))

    panel_w = 260
    panel_h = 200
    gap_x = 40
    gap_y = 50
    margin = {"left": 60, "right": 20, "top": 50, "bottom": 40}

    width = margin["left"] + margin["right"] + cols * panel_w + (cols - 1) * gap_x
    height = margin["top"] + margin["bottom"] + rows * panel_h + (rows - 1) * gap_y

    betas = sorted({p.beta for s in series_list for p in s.points})
    x_min, x_max = min(betas), max(betas)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    x_pad = 0.1 * (x_max - x_min)
    x_min -= x_pad
    x_max += x_pad

    values = [value_fn(p) for s in series_list for p in s.points]
    y_min = min(values)
    y_max = max(values)
    if y_min == y_max:
        y_min -= 0.1
        y_max += 0.1
    y_pad = 0.12 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
        f'font-family="Arial" font-size="18" fill="#222">{title}</text>'
    )

    inner = {"left": 40, "right": 10, "top": 20, "bottom": 30}
    xticks = _ticks(x_min, x_max)
    yticks = _ticks(y_min, y_max)

    for idx, cfg in enumerate(series_list):
        row = idx // cols
        col = idx % cols
        panel_x = margin["left"] + col * (panel_w + gap_x)
        panel_y = margin["top"] + row * (panel_h + gap_y)

        plot_x0 = panel_x + inner["left"]
        plot_y0 = panel_y + inner["top"]
        plot_w = panel_w - inner["left"] - inner["right"]
        plot_h = panel_h - inner["top"] - inner["bottom"]

        def x_px(x: float) -> float:
            return plot_x0 + (x - x_min) / (x_max - x_min) * plot_w

        def y_px(y: float) -> float:
            return plot_y0 + (y_max - y) / (y_max - y_min) * plot_h

        parts.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#e0e0e0"/>')
        parts.append(f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="#222"/>')
        parts.append(f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="#222"/>')

        for xv in xticks:
            xp = x_px(xv)
            parts.append(f'<line x1="{xp:.1f}" y1="{plot_y0 + plot_h}" x2="{xp:.1f}" y2="{plot_y0 + plot_h + 4}" stroke="#222"/>')
            if row == rows - 1:
                parts.append(
                    f'<text x="{xp:.1f}" y="{plot_y0 + plot_h + 16}" text-anchor="middle" '
                    f'font-family="Arial" font-size="10" fill="#222">{xv:.2f}</text>'
                )
        for yv in yticks:
            yp = y_px(yv)
            parts.append(f'<line x1="{plot_x0 - 4}" y1="{yp:.1f}" x2="{plot_x0}" y2="{yp:.1f}" stroke="#222"/>')
            if col == 0:
                parts.append(
                    f'<text x="{plot_x0 - 6}" y="{yp + 3:.1f}" text-anchor="end" '
                    f'font-family="Arial" font-size="10" fill="#222">{yv:.3f}</text>'
                )

        label = f"{cfg.n_qubits}q d{cfg.depth}"
        parts.append(
            f'<text x="{panel_x + 8}" y="{panel_y + 14}" text-anchor="start" '
            f'font-family="Arial" font-size="12" fill="#222">{label}</text>'
        )
        if row == rows - 1:
            parts.append(
                f'<text x="{plot_x0 + plot_w/2:.1f}" y="{panel_y + panel_h - 6}" '
                f'text-anchor="middle" font-family="Arial" font-size="11" fill="#222">beta</text>'
            )
        if col == 0:
            parts.append(
                f'<text x="{panel_x + 10}" y="{plot_y0 + plot_h/2:.1f}" text-anchor="middle" '
                f'font-family="Arial" font-size="11" fill="#222" '
                f'transform="rotate(-90 {panel_x + 10} {plot_y0 + plot_h/2:.1f})">{y_label}</text>'
            )

        points = sorted(cfg.points, key=lambda p: p.beta)
        xs = [x_px(p.beta) for p in points]
        ys = [y_px(value_fn(p)) for p in points]
        color = colors[idx % len(colors)]

        line = " ".join([f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)])
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>')

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_fixed_lines_svg(
    series_list: List[FixedSeries],
    value_fn: Callable[[FixedPoint], float],
    y_label: str,
    out_path: str,
    title: str,
) -> None:
    width, height = 800, 480
    margin = {"left": 70, "right": 160, "top": 50, "bottom": 60}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    betas = sorted({p.beta for s in series_list for p in s.points})
    x_min, x_max = min(betas), max(betas)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    x_pad = 0.1 * (x_max - x_min)
    x_min -= x_pad
    x_max += x_pad

    values = [value_fn(p) for s in series_list for p in s.points]
    y_min = min(values)
    y_max = max(values)
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
        f'<text x="{x0 + plot_w/2:.1f}" y="{height-20}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222">beta</text>'
    )
    parts.append(
        f'<text x="20" y="{margin["top"] + plot_h/2:.1f}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222" '
        f'transform="rotate(-90 20 {margin["top"] + plot_h/2:.1f})">{y_label}</text>'
    )

    for idx, series in enumerate(series_list):
        color = colors[idx % len(colors)]
        points = sorted(series.points, key=lambda p: p.beta)
        xs = [x_px(p.beta) for p in points]
        ys = [y_px(value_fn(p)) for p in points]
        line = " ".join([f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys)])
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for x, y in zip(xs, ys):
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')

    legend_x = width - margin["right"] + 20
    legend_y = margin["top"] + 10
    parts.append(
        f'<rect x="{legend_x}" y="{legend_y - 16}" width="130" height="{18 * len(series_list) + 10}" '
        f'fill="#f7f7f7" stroke="#ddd"/>'
    )
    for idx, series in enumerate(series_list):
        color = colors[idx % len(colors)]
        y = legend_y + idx * 18
        label = f"n={series.n_samples}"
        parts.append(f'<rect x="{legend_x + 8}" y="{y - 9}" width="10" height="10" fill="{color}"/>')
        parts.append(
            f'<text x="{legend_x + 24}" y="{y}" font-family="Arial" font-size="12" fill="#222">'
            f'{label}</text>'
        )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

def run_summary(
    configs: List[Tuple[int, int]],
    betas: List[float],
    n_samples: int,
    n_trials: int,
    seed: int,
    p_I_range,
) -> List[SummarySeries]:
    print("\nSUMMARY (random instances)")
    print("config, beta, mean_gamma, mean_std, mean_bias, rmse")
    summary_series = []
    for n_qubits, depth in configs:
        stats: Dict[float, Dict[str, List[float]]] = {
            b: {"gamma": [], "std": [], "bias": []} for b in betas
        }
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
                stats[beta]["gamma"].append(float(est.gamma))
                stats[beta]["std"].append(float(est.std))
                stats[beta]["bias"].append(float(est.mean - ideal))

        for beta in betas:
            gammas = np.array(stats[beta]["gamma"])
            stds = np.array(stats[beta]["std"])
            biases = np.array(stats[beta]["bias"])
            mean_gamma = float(gammas.mean())
            mean_std = float(stds.mean())
            mean_bias = float(biases.mean())
            rmse = float(np.sqrt(np.mean(biases ** 2)))
            label = f"{n_qubits}q_d{depth}"
            print(f"{label}, {beta:.4f}, {mean_gamma:.3f}, {mean_std:.6f}, {mean_bias:.6f}, {rmse:.6f}")
        points = [
            SummaryPoint(
                beta=b,
                mean_gamma=float(np.mean(stats[b]["gamma"])),
                mean_std=float(np.mean(stats[b]["std"])),
                mean_bias=float(np.mean(stats[b]["bias"])),
            )
            for b in betas
        ]
        summary_series.append(SummarySeries(n_qubits=n_qubits, depth=depth, points=points))
    return summary_series


def run_fixed_instance(
    n_qubits: int,
    depth: int,
    betas: List[float],
    n_samples_list: List[int],
    n_repeats: int,
    seed: int,
    p_I_range,
) -> List[FixedSeries]:
    rng = np.random.default_rng(seed)
    circuit = random_circuit(n_qubits, depth, rng)
    noise = random_noise_model(rng, p_I_range=p_I_range)
    init = random_product_state(n_qubits, rng)
    obs = random_observable(n_qubits, rng)
    locs = error_locations(circuit, noise)

    sim = NoisySimulator(STANDARD_GATES, noise)
    ideal = sim.ideal(circuit, obs, init)

    print("\nFIXED INSTANCE")
    print(f"config={n_qubits}q_d{depth}, seed={seed}")
    print("n_samples, beta, mean_gamma, mean_bias, emp_std, mean_est_std, emp_rmse")

    fixed_series = []
    for n_samples in n_samples_list:
        points = []
        for beta in betas:
            ests = []
            stds = []
            gammas = []
            for r in range(n_repeats):
                est = pec_estimate_qiskit(
                    circuit,
                    obs,
                    init,
                    noise,
                    locs,
                    beta=beta,
                    n_samples=n_samples,
                    seed=seed + 2000 * r + int(100 * beta),
                )
                ests.append(float(est.mean))
                stds.append(float(est.std))
                gammas.append(float(est.gamma))

            ests_arr = np.array(ests)
            errors = ests_arr - ideal
            mean_gamma = float(np.mean(gammas))
            mean_bias = float(np.mean(errors))
            emp_std = float(np.std(errors, ddof=1)) if len(errors) > 1 else 0.0
            mean_est_std = float(np.mean(stds))
            emp_rmse = float(np.sqrt(np.mean(errors ** 2)))

            print(
                f"{n_samples}, {beta:.4f}, {mean_gamma:.3f}, "
                f"{mean_bias:.6f}, {emp_std:.6f}, {mean_est_std:.6f}, {emp_rmse:.6f}"
            )
            points.append(
                FixedPoint(
                    beta=beta,
                    mean_gamma=mean_gamma,
                    mean_bias=mean_bias,
                    emp_std=emp_std,
                    mean_est_std=mean_est_std,
                )
            )
        fixed_series.append(FixedSeries(n_samples=n_samples, points=points))
    return fixed_series


def main() -> None:
    parser = argparse.ArgumentParser(description="Variance checks for exponential-window PEC.")
    parser.add_argument("--betas", default="0.00,0.05,0.10,0.20")
    parser.add_argument("--configs", default="3x4,4x4,5x4")
    parser.add_argument("--n-samples", default="40,80,160")
    parser.add_argument("--n-trials", type=int, default=12)
    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--noise-profile",
        choices=["uniform", "gate"],
        default="gate",
        help="Noise profile: uniform p_I range or gate-dependent ranges.",
    )
    parser.add_argument("--fixed-config", default="")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument("--out-prefix", default="beta_variance")
    parser.add_argument("--cols", type=int, default=3)
    args = parser.parse_args()

    betas = parse_betas(args.betas)
    configs = parse_configs(args.configs)
    n_samples_list = parse_int_list(args.n_samples)
    p_I_range = (0.85, 0.95) if args.noise_profile == "uniform" else GATE_P_I_RANGES

    summary_series = []
    fixed_series = []

    if not args.skip_summary:
        summary_series = run_summary(
            configs=configs,
            betas=betas,
            n_samples=n_samples_list[0],
            n_trials=args.n_trials,
            seed=args.seed,
            p_I_range=p_I_range,
        )

    if not args.skip_fixed:
        if args.fixed_config:
            fixed_cfg = parse_configs(args.fixed_config)
            n_qubits, depth = fixed_cfg[0]
        else:
            n_qubits, depth = configs[0]

        fixed_series = run_fixed_instance(
            n_qubits=n_qubits,
            depth=depth,
            betas=betas,
            n_samples_list=n_samples_list,
            n_repeats=args.n_repeats,
            seed=args.seed,
            p_I_range=p_I_range,
        )

    if summary_series:
        title = (
            f"Summary (n_samples={n_samples_list[0]}, n_trials={args.n_trials}, "
            f"noise={args.noise_profile})"
        )
        write_facets_metric_svg(
            summary_series,
            lambda p: p.mean_gamma,
            "Mean gamma",
            f"{args.out_prefix}_summary_gamma.svg",
            title,
            cols=args.cols,
        )
        write_facets_metric_svg(
            summary_series,
            lambda p: p.mean_std,
            "Mean std",
            f"{args.out_prefix}_summary_std.svg",
            title,
            cols=args.cols,
        )
        write_facets_metric_svg(
            summary_series,
            lambda p: p.mean_bias,
            "Mean bias",
            f"{args.out_prefix}_summary_bias.svg",
            title,
            cols=args.cols,
        )
        print(
            f"Wrote {args.out_prefix}_summary_gamma.svg, "
            f"{args.out_prefix}_summary_std.svg, "
            f"{args.out_prefix}_summary_bias.svg"
        )

    if fixed_series:
        title = (
            f"Fixed instance (config={n_qubits}q_d{depth}, repeats={args.n_repeats}, "
            f"noise={args.noise_profile})"
        )
        write_fixed_lines_svg(
            fixed_series,
            lambda p: p.emp_std,
            "Empirical std",
            f"{args.out_prefix}_fixed_std.svg",
            title,
        )
        write_fixed_lines_svg(
            fixed_series,
            lambda p: p.mean_bias,
            "Mean bias",
            f"{args.out_prefix}_fixed_bias.svg",
            title,
        )
        print(
            f"Wrote {args.out_prefix}_fixed_std.svg, "
            f"{args.out_prefix}_fixed_bias.svg"
        )


if __name__ == "__main__":
    main()
