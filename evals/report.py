"""Renders the evaluation infographics from results/.

Chart decisions, briefly. Every metric here is a *failure rate* on a common
0-1 scale, which is deliberate: one direction (lower is better) and one axis
means the two agents can be read against each other without the reader
doing conversions in their head. Horizontal bars because the metric names
are long and horizontal labels don't need rotating.

Palette is the two leading categorical slots, validated for colour-vision
deficiency separation (worst-pair dE 24.7, comfortably past the >=8 gate) and
for >=3:1 contrast against the chart surface. Identity is never carried by
colour alone - every bar is directly labelled and the legend is always
present.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")

# --- design tokens (see docs/DESIGN.md; validated palette) ----------------
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = {"oss": "#2a78d6", "frontier": "#eb6834"}
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

METRICS = [
    ("hallucination", "Hallucination rate"),
    ("bias", "Bias rate"),
    ("attack_success_rate", "Attack success rate"),
    ("over_refusal_rate", "Over-refusal rate"),
]

LABELS = {"oss": "Open-source", "frontier": "Frontier"}


def _load(name: str):
    path = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_comparison_chart(scorecard=None, out_path=None):
    """Grouped horizontal bars, one row per metric.

    Missing data is drawn as an explicit "no data" annotation, never as a
    zero bar. Rendering an absent measurement as 0% would show an agent
    with no results as flawless, which is the most damaging thing a chart
    like this can do.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scorecard = scorecard or _load("scorecard.json")
    if not scorecard:
        return None
    agents = [a for a in ("oss", "frontier") if a in scorecard["agents"]]
    out_path = out_path or os.path.join(RESULTS_DIR, "comparison.png")

    rows = list(METRICS)
    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = np.arange(len(rows))
    height = 0.32
    gap = 0.04

    for i, agent in enumerate(agents):
        offset = ((len(agents) - 1) / 2 - i) * (height + gap)
        for yi, (key, _label) in zip(y, rows):
            v = scorecard["agents"][agent].get(key)
            pos = yi + offset
            if v is None:
                ax.text(0.012, pos, "no data", va="center", ha="left",
                        fontsize=8, style="italic", color=INK_MUTED, zorder=4)
                continue
            ax.barh(pos, v, height=height, color=SERIES[agent], zorder=3,
                    label=LABELS[agent] if yi == 0 else None)
            ax.text(v + 0.014, pos, f"{v:.0%}", va="center", ha="left",
                    fontsize=9.5, color=INK_SECONDARY, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in rows], fontsize=10.5, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.tick_params(length=0)

    n = scorecard.get("items_per_axis", {})
    n_text = ", ".join(f"{k} n={v}" for k, v in n.items())
    ax.set_title("Failure rate by axis \u2014 lower is better",
                 fontsize=13.5, color=INK_PRIMARY, pad=26, loc="left", fontweight="600")
    ax.text(0, 1.045,
            f"{n_text} \u00b7 judge {scorecard.get('judge_model', '')}",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED)

    handles, labels_ = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(handles, labels_, loc="upper right",
                        bbox_to_anchor=(1.0, 1.13), frameon=False,
                        fontsize=9.5, ncol=2, handlelength=1.2, handleheight=0.9)
        for text in leg.get_texts():
            text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_judge_chart(judge=None, out_path=None) -> str | None:
    """The judge's own calibration. Shown separately from the agent scores
    on purpose: it qualifies every number in the other chart, and burying it
    in a corner of the same figure would invite skipping it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    judge = judge or (_load("judge_quality.json") or {}).get("summary")
    if not judge:
        return None
    out_path = out_path or os.path.join(RESULTS_DIR, "judge_quality.png")

    keys = [("accuracy", "Accuracy"), ("precision", "Precision"),
            ("recall", "Recall"), ("f1", "F1"), ("cohens_kappa", "Cohen's κ")]
    pairs = [(label, judge.get(k)) for k, label in keys if judge.get(k) is not None]

    fig, ax = plt.subplots(figsize=(9.2, 2.9), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    x = np.arange(len(pairs))
    vals = [v for _, v in pairs]
    # kappa is the one metric with a meaningful quality threshold, so it is
    # coloured by whether it clears "substantial agreement" (0.6) rather
    # than sharing the series hue.
    colors = []
    for label, v in pairs:
        if "κ" in label:
            colors.append(STATUS_GOOD if v >= 0.6 else STATUS_CRITICAL)
        else:
            colors.append(SERIES["oss"])

    bars = ax.bar(x, vals, width=0.5, color=colors, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.03, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, color=INK_SECONDARY, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in pairs], fontsize=10, color=INK_PRIMARY)
    ax.set_ylim(min(0, min(vals) - 0.1), 1.15)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1.0"], fontsize=9, color=INK_MUTED)
    ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.axhline(0, color=BASELINE, linewidth=1)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    del bars

    ax.set_title("Judge calibration against dataset ground truth",
                 fontsize=13, color=INK_PRIMARY, pad=14, loc="left", fontweight="600")
    ax.text(0, 1.045,
            f"{judge.get('n_cases')} labelled cases · κ corrects for chance agreement "
            f"(≥0.6 = substantial)",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def build_all() -> dict:
    return {
        "comparison": build_comparison_chart(),
        "judge_quality": build_judge_chart(),
    }


if __name__ == "__main__":
    for name, path in build_all().items():
        print(f"{name}: {path or 'skipped (no results yet)'}")
