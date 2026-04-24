#!/usr/bin/env python
"""Overlay our reproduced training curve on top of FinePhrase Figure 1.

Reads the `eval_curve/curve.json` produced by `eval_all_ckpts.py` and plots
our macro-aggregate score vs training tokens, alongside the published
FinePhrase Figure 1 baselines (transcribed from the screenshots).

Usage:
    python scripts/plot_figure1_overlay.py \
        --curve experiments/exp_finephrase_repro_protC_seed42/eval_curve/curve.json \
        --out figures/figure1_overlay.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Hand-transcribed from FinePhrase Figure 1 (Aggregate Score Macro view).
# Each entry = (token_count_billions, score). Step ~1B between points.
FINEPHRASE_FIG1_BASELINES: dict[str, list[tuple[float, float]]] = {
    "FinePhrase (table)": [
        (0.5, 0.062), (1.5, 0.090), (2.5, 0.103), (3.5, 0.116),
        (4.5, 0.124), (5.5, 0.131), (6.5, 0.137), (7.5, 0.137),
        (8.5, 0.140), (9.5, 0.144), (10.5, 0.144), (11.5, 0.151),
        (12.5, 0.146), (13.5, 0.155), (14.5, 0.146), (15.5, 0.157),
        (16.5, 0.158), (17.5, 0.158), (18.5, 0.160), (19.5, 0.158),
        (20.5, 0.169), (21.0, 0.171),
    ],
    "Nemotron-HQ-Synth": [
        (0.5, 0.052), (2.5, 0.080), (4.5, 0.097), (6.5, 0.105),
        (8.5, 0.110), (10.5, 0.117), (12.5, 0.121), (14.5, 0.123),
        (16.5, 0.125), (18.5, 0.130), (20.5, 0.135), (21.0, 0.135),
    ],
    "REWIRE": [
        (0.5, 0.048), (2.5, 0.080), (4.5, 0.095), (6.5, 0.103),
        (8.5, 0.110), (10.5, 0.115), (12.5, 0.118), (14.5, 0.121),
        (16.5, 0.123), (18.5, 0.128), (20.5, 0.135), (21.0, 0.135),
    ],
    "Cosmopedia": [
        (0.5, 0.034), (2.5, 0.058), (4.5, 0.066), (6.5, 0.072),
        (8.5, 0.080), (10.5, 0.083), (12.5, 0.087), (14.5, 0.091),
        (16.5, 0.094), (18.5, 0.098), (20.5, 0.102), (21.0, 0.103),
    ],
    "SYNTH": [
        (0.5, 0.038), (2.5, 0.064), (4.5, 0.073), (6.5, 0.078),
        (8.5, 0.085), (10.5, 0.089), (12.5, 0.092), (14.5, 0.094),
        (16.5, 0.097), (18.5, 0.099), (20.5, 0.100), (21.0, 0.100),
    ],
}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--curve", type=Path, required=True, help="path to eval_curve/curve.json")
    p.add_argument("--out", type=Path, required=True, help="output PNG path")
    p.add_argument("--label", default="Incepedia (Protocol C reproduction)")
    p.add_argument("--macro-key", default="macro",
                   help="key in each curve row whose value is the y-coord (default: macro)")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib", file=sys.stderr)
        return 1

    curve = json.loads(args.curve.read_text())
    rows = curve["rows"]
    gbt = curve["global_batch_tokens"]
    xs_b = [r["step"] * gbt / 1e9 for r in rows]
    ys = [r[args.macro_key] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))

    # FinePhrase baselines (transcribed)
    colors = {
        "FinePhrase (table)": "#FF9F1C",
        "Nemotron-HQ-Synth": "#2EC4B6",
        "REWIRE": "#3A86FF",
        "Cosmopedia": "#E63946",
        "SYNTH": "#9D4EDD",
    }
    for name, pts in FINEPHRASE_FIG1_BASELINES.items():
        bx = [t for (t, _) in pts]
        by = [s for (_, s) in pts]
        ax.plot(bx, by, "-o", linewidth=1.5, markersize=3, alpha=0.85,
                color=colors.get(name, "gray"), label=name)

    # Our curve
    ax.plot(xs_b, ys, "-D", linewidth=2.5, markersize=6,
            color="#0B5394", label=args.label)

    ax.set_xlabel("Tokens (Steps)", fontsize=12)
    ax.set_ylabel(f"{args.macro_key.replace('_', ' ').title()}", fontsize=12)
    ax.set_xlim(0, 22)
    ax.set_ylim(0, max(0.20, max(ys) * 1.1 if ys else 0.20))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Training Progression — Incepedia overlay on FinePhrase Figure 1",
                 fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"[plot] wrote {args.out}")
    print(f"[plot] our final macro: {ys[-1]:.4f} at {xs_b[-1]:.1f}B tokens")
    print(f"[plot] FinePhrase (table) final: {FINEPHRASE_FIG1_BASELINES['FinePhrase (table)'][-1][1]:.4f}")
    delta = ys[-1] - FINEPHRASE_FIG1_BASELINES["FinePhrase (table)"][-1][1]
    sign = "+" if delta >= 0 else ""
    print(f"[plot] delta vs FinePhrase: {sign}{delta:.4f} ({sign}{delta*100:.2f}pp)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
