"""Builds results/chart.png (grouped bar chart, one group per axis, one bar
per agent) from results/scorecard.json - the infographic for the 1-page
evaluation report (plan.md Section 5 / deliverable #3).
"""
from __future__ import annotations

import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def build_chart(scorecard_path: str | None = None, out_path: str | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scorecard_path = scorecard_path or os.path.join(RESULTS_DIR, "scorecard.json")
    out_path = out_path or os.path.join(RESULTS_DIR, "chart.png")

    with open(scorecard_path, encoding="utf-8") as f:
        scorecard = json.load(f)

    agents = list(scorecard.keys())
    axes = sorted({axis for agent_scores in scorecard.values() for axis in agent_scores})

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.8 / max(len(agents), 1)
    x = range(len(axes))

    for i, agent in enumerate(agents):
        values = [scorecard[agent].get(axis) or 0 for axis in axes]
        positions = [xi + i * width for xi in x]
        ax.bar(positions, values, width=width, label=agent)

    ax.set_xticks([xi + width * (len(agents) - 1) / 2 for xi in x])
    ax.set_xticklabels(axes, rotation=20, ha="right")
    ax.set_ylabel("Score (lower is better for hallucination/bias/attack-success)")
    ax.set_title("Wellness Assistant: OSS vs. Frontier — Eval Scorecard")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    return out_path


if __name__ == "__main__":
    print(build_chart())
