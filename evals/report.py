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


def _rounded_bars(ax, patches, radius_frac=0.45):
    """matplotlib has no rounded bar ends, so each bar is redrawn as a
    rounded patch. Purely cosmetic - it makes short bars read as marks
    rather than as slivers of the axis."""
    from matplotlib.patches import FancyBboxPatch

    for p in patches:
        x, y = p.get_xy()
        w, h = p.get_width(), p.get_height()
        if w <= 0:
            continue
        p.set_visible(False)
        r = min(h * radius_frac, w / 2)
        ax.add_patch(FancyBboxPatch(
            (x, y), max(w - r, 0.0001), h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            facecolor=p.get_facecolor(), edgecolor="none",
            mutation_aspect=1, zorder=3, clip_on=False))


def build_comparison_chart(scorecard=None, out_path=None) -> str | None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scorecard = scorecard or _load("scorecard.json")
    if not scorecard:
        return None
    agents = [a for a in ("oss", "frontier") if a in scorecard["agents"]]
    out_path = out_path or os.path.join(RESULTS_DIR, "comparison.png")

    rows = [(key, label) for key, label in METRICS
            if any(scorecard["agents"][a].get(key) is not None for a in agents)]

    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    y = np.arange(len(rows))
    height = 0.34
    gap = 0.02  # 2px-equivalent surface gap between adjacent bars

    for i, agent in enumerate(agents):
        offset = (i - (len(agents) - 1) / 2) * (height + gap)
        vals = [scorecard["agents"][agent].get(k) or 0.0 for k, _ in rows]
        bars = ax.barh(y - offset, vals, height=height,
                       color=SERIES[agent], label=LABELS[agent], zorder=3)
        _rounded_bars(ax, bars)
        for yi, v in zip(y - offset, vals):
            ax.text(v + 0.018, yi, f"{v:.0%}", va="center", ha="left",
                    fontsize=9, color=INK_SECONDARY, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels([label for _, label in rows], fontsize=10.5, color=INK_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=9, color=INK_MUTED)
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.tick_params(length=0)

    ax.set_title("Failure rate by axis — lower is better",
                 fontsize=13, color=INK_PRIMARY, pad=14, loc="left", fontweight="600")
    ax.text(0, 1.055, f"{scorecard['items_per_axis']} items per axis · "
                      f"judge: {scorecard['judge_model']}",
            transform=ax.transAxes, fontsize=8.5, color=INK_MUTED)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9.5, ncol=2)
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
