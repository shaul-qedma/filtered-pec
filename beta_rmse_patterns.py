"""
Pattern plots for RMSE vs beta across multiple configs.
Produces a facet grid with CI ribbons and a heatmap summary.
"""

import argparse
import math
import time
from typing import List, Tuple

from beta_rmse_scatter import (
    GATE_P_I_RANGES,
    BetaRMSE,
    ConfigRMSE,
    parse_betas,
    parse_configs,
    run_trials_for_config,
)


def _ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if vmax == vmin:
        return [vmin]
    step = (vmax - vmin) / (count - 1)
    return [vmin + i * step for i in range(count)]


def _hex(c: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*c)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _color_scale(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        return _hex((200, 200, 200))
    t = (value - vmin) / (vmax - vmin)
    t = max(0.0, min(1.0, t))
    start = (33, 102, 172)  # blue
    end = (230, 85, 13)     # orange
    r = int(_lerp(start[0], end[0], t))
    g = int(_lerp(start[1], end[1], t))
    b = int(_lerp(start[2], end[2], t))
    return _hex((r, g, b))


def parse_int_list(raw: str) -> List[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def estimate_error_locs(n_qubits: int, depth: int) -> int:
    """Estimate number of noisy locations for brickwall circuits."""
    twoq_layers = depth // 2
    gates_per_layer = n_qubits // 2
    return 2 * gates_per_layer * twoq_layers


def scaled_samples(
    base: int,
    n_locs: int,
    min_locs: int,
    mode: str,
    max_samples: int,
) -> int:
    if mode == "none":
        return base

    if mode == "locs":
        denom = max(1, min_locs)
        factor = max(1.0, n_locs / denom)
    else:
        delta = max(0, n_locs - min_locs)
        if max_samples:
            if delta > 0:
                log_target = math.log(max_samples)
                if math.log(base) + delta * math.log(4) >= log_target:
                    return max_samples
        factor = 4 ** delta

    scaled = int(math.ceil(base * factor))
    if max_samples and scaled > max_samples:
        return max_samples
    return scaled


def write_facets_svg(
    results: List[ConfigRMSE],
    out_path: str,
    title: str,
    cols: int,
) -> None:
    n_cfg = len(results)
    cols = max(1, min(cols, n_cfg))
    rows = int(math.ceil(n_cfg / cols))

    panel_w = 260
    panel_h = 200
    gap_x = 40
    gap_y = 50
    margin = {"left": 60, "right": 20, "top": 50, "bottom": 40}

    width = margin["left"] + margin["right"] + cols * panel_w + (cols - 1) * gap_x
    height = margin["top"] + margin["bottom"] + rows * panel_h + (rows - 1) * gap_y

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

    for idx, cfg in enumerate(results):
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

        # Axes
        parts.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="none" stroke="#e0e0e0"/>')
        parts.append(f'<line x1="{plot_x0}" y1="{plot_y0 + plot_h}" x2="{plot_x0 + plot_w}" y2="{plot_y0 + plot_h}" stroke="#222"/>')
        parts.append(f'<line x1="{plot_x0}" y1="{plot_y0}" x2="{plot_x0}" y2="{plot_y0 + plot_h}" stroke="#222"/>')

        # Ticks
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

        # Labels
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
                f'transform="rotate(-90 {panel_x + 10} {plot_y0 + plot_h/2:.1f})">RMSE</text>'
            )

        # CI ribbon and line
        stats = sorted(cfg.stats, key=lambda s: s.beta)
        xs = [x_px(s.beta) for s in stats]
        ys_hi = [y_px(s.ci_high) for s in stats]
        ys_lo = [y_px(s.ci_low) for s in stats]
        ribbon = " ".join(
            [f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys_hi)]
            + [f"{x:.1f},{y:.1f}" for x, y in zip(reversed(xs), reversed(ys_lo))]
        )
        color = colors[idx % len(colors)]
        parts.append(f'<polygon points="{ribbon}" fill="{color}" opacity="0.18" stroke="none"/>')

        line = " ".join(
            [f"{x:.1f},{y_px(s.rmse):.1f}" for x, s in zip(xs, stats)]
        )
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for x, s in zip(xs, stats):
            parts.append(f'<circle cx="{x:.1f}" cy="{y_px(s.rmse):.1f}" r="3.5" fill="{color}"/>')

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def write_heatmap_svg(
    results: List[ConfigRMSE],
    out_path: str,
    title: str,
) -> None:
    width, height = 900, 420
    margin = {"left": 120, "right": 80, "top": 50, "bottom": 70}
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]

    betas = sorted({s.beta for r in results for s in r.stats})
    n_beta = len(betas)
    n_cfg = len(results)

    all_rmse = [s.rmse for r in results for s in r.stats]
    vmin, vmax = min(all_rmse), max(all_rmse)
    cell_w = plot_w / n_beta
    cell_h = plot_h / n_cfg

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{width/2:.1f}" y="28" text-anchor="middle" '
        f'font-family="Arial" font-size="18" fill="#222">{title}</text>'
    )

    # Cells
    for i, cfg in enumerate(results):
        y = margin["top"] + i * cell_h
        stats = {s.beta: s for s in cfg.stats}
        for j, beta in enumerate(betas):
            x = margin["left"] + j * cell_w
            rmse = stats[beta].rmse
            color = _color_scale(rmse, vmin, vmax)
            parts.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_w:.1f}" height="{cell_h:.1f}" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1"/>'
            )

    # Axes labels
    for j, beta in enumerate(betas):
        x = margin["left"] + j * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{height - 40}" text-anchor="middle" '
            f'font-family="Arial" font-size="12" fill="#222">{beta:.2f}</text>'
        )
    for i, cfg in enumerate(results):
        y = margin["top"] + i * cell_h + cell_h / 2
        label = f"{cfg.n_qubits}q d{cfg.depth}"
        parts.append(
            f'<text x="{margin["left"] - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="Arial" font-size="12" fill="#222">{label}</text>'
        )

    parts.append(
        f'<text x="{margin["left"] + plot_w/2:.1f}" y="{height - 18}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222">beta</text>'
    )
    parts.append(
        f'<text x="20" y="{margin["top"] + plot_h/2:.1f}" text-anchor="middle" '
        f'font-family="Arial" font-size="14" fill="#222" '
        f'transform="rotate(-90 20 {margin["top"] + plot_h/2:.1f})">config</text>'
    )

    # Legend
    legend_x = width - margin["right"] + 20
    legend_y = margin["top"] + 20
    legend_h = plot_h - 40
    steps = 40
    step_h = legend_h / steps
    for i in range(steps):
        t = i / (steps - 1)
        value = _lerp(vmax, vmin, t)
        color = _color_scale(value, vmin, vmax)
        y = legend_y + i * step_h
        parts.append(
            f'<rect x="{legend_x}" y="{y:.1f}" width="16" height="{step_h + 1:.1f}" fill="{color}" stroke="none"/>'
        )
    parts.append(
        f'<text x="{legend_x + 22}" y="{legend_y + 8}" font-family="Arial" font-size="11" fill="#222">{vmax:.3f}</text>'
    )
    parts.append(
        f'<text x="{legend_x + 22}" y="{legend_y + legend_h}" font-family="Arial" font-size="11" fill="#222">{vmin:.3f}</text>'
    )
    parts.append(
        f'<text x="{legend_x + 8}" y="{legend_y - 8}" text-anchor="middle" '
        f'font-family="Arial" font-size="11" fill="#222">RMSE</text>'
    )

    parts.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Facet + heatmap plots for RMSE vs beta.")
    parser.add_argument("--betas", default="0.00,0.05,0.10,0.20", help="Comma-separated beta values.")
    parser.add_argument(
        "--configs",
        default="2x3,2x5,3x4,3x6,4x4,4x6,5x4,5x6,6x4,6x5",
        help="Comma-separated configs like 3x2.",
    )
    parser.add_argument(
        "--n-samples",
        default="40,80,160",
        help="Comma-separated sample counts to test scaling.",
    )
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument(
        "--noise-profile",
        choices=["uniform", "gate"],
        default="gate",
        help="Noise profile: uniform p_I range or gate-dependent ranges.",
    )
    parser.add_argument(
        "--scale-mode",
        choices=["space", "locs", "none"],
        default="space",
        help="Scale samples by configuration space size (4^n_locs) or by n_locs.",
    )
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--out-prefix", default="beta_rmse")
    parser.add_argument("--out-facets", default="beta_rmse_facets.svg")
    parser.add_argument("--out-heatmap", default="beta_rmse_heatmap.svg")
    args = parser.parse_args()

    betas = parse_betas(args.betas)
    configs = parse_configs(args.configs)
    n_samples_list = parse_int_list(args.n_samples)
    p_I_range = (0.85, 0.95) if args.noise_profile == "uniform" else GATE_P_I_RANGES
    n_locs_map = {cfg: estimate_error_locs(*cfg) for cfg in configs}
    min_locs = min(n_locs_map.values()) if n_locs_map else 0

    t0 = time.time()
    for n_samples in n_samples_list:
        results = []
        for n_qubits, depth in configs:
            n_locs = n_locs_map[(n_qubits, depth)]
            scaled_n_samples = scaled_samples(
                base=n_samples,
                n_locs=n_locs,
                min_locs=min_locs,
                mode=args.scale_mode,
                max_samples=args.max_samples,
            )
            results.append(
                run_trials_for_config(
                    n_qubits=n_qubits,
                    depth=depth,
                    betas=betas,
                    n_samples=scaled_n_samples,
                    n_trials=args.n_trials,
                    seed=args.seed,
                    confidence=args.confidence,
                    n_boot=args.bootstrap,
                    p_I_range=p_I_range,
                )
            )

        print(
            f"sample scaling: base={n_samples}, mode={args.scale_mode}, "
            f"max={args.max_samples}, min_locs={min_locs}"
        )
        print("config, beta, rmse, ci_low, ci_high")
        for cfg in results:
            label = f"{cfg.n_qubits}q_d{cfg.depth}"
            for s in cfg.stats:
                print(f"{label}, {s.beta:.4f}, {s.rmse:.6f}, {s.ci_low:.6f}, {s.ci_high:.6f}")

        title = (
            f"RMSE vs beta (base={n_samples}, scale={args.scale_mode}, "
            f"n_trials={args.n_trials}, noise={args.noise_profile})"
        )

        if len(n_samples_list) == 1:
            out_facets = args.out_facets
            out_heatmap = args.out_heatmap
        else:
            out_facets = f"{args.out_prefix}_facets_n{n_samples}.svg"
            out_heatmap = f"{args.out_prefix}_heatmap_n{n_samples}.svg"

        write_facets_svg(results, out_facets, title, cols=args.cols)
        write_heatmap_svg(results, out_heatmap, title)
        print(f"Wrote {out_facets} and {out_heatmap}")

    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
