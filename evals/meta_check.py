"""Judge meta-check (plan.md Section 6): sample 5 items/axis, hand-label
them yourself, compare to the judge's verdict, report a simple % agreement.
This is what satisfies the brief's "assess the quality of the judge"
requirement at 1-day-POC depth (no formal Cohen's kappa, single labeler -
documented as a known limitation, not hidden).

Usage:
    1. Run `python -m evals.meta_check sample` to write
       evals/gold_labels_template.jsonl with 15 sampled items (5/axis).
    2. Fill in the "human_label" field by hand for each line
       (hallucinated/not_hallucinated, biased/not_biased, unsafe/safe) and
       save as evals/gold_labels.jsonl.
    3. Run `python -m evals.meta_check score` to compare against the judge
       and print the agreement %.
"""
from __future__ import annotations

import json
import os
import random
import sys

from evals.runner import DATASETS_DIR, load_jsonl

GOLD_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "gold_labels_template.jsonl")
GOLD_PATH = os.path.join(os.path.dirname(__file__), "gold_labels.jsonl")

random.seed(1)


def build_sample(n_per_axis: int = 5) -> list[dict]:
    halluc = load_jsonl(os.path.join(DATASETS_DIR, "hallucination.jsonl"))
    bias = load_jsonl(os.path.join(DATASETS_DIR, "bias.jsonl"))
    safety = load_jsonl(os.path.join(DATASETS_DIR, "safety.jsonl"))

    sample = []
    for axis, items, _text_field in (
        ("hallucination", halluc, "question"),
        ("bias", bias, "question"),
        ("safety", safety, "prompt"),
    ):
        picked = random.sample(items, min(n_per_axis, len(items)))
        for item in picked:
            sample.append({
                "axis": axis,
                "item": item,
                "human_label": None,  # fill by hand: e.g. "hallucinated"/"not_hallucinated"
            })
    return sample


def write_template(n_per_axis: int = 5) -> str:
    sample = build_sample(n_per_axis)
    with open(GOLD_TEMPLATE_PATH, "w", encoding="utf-8") as f:
        for row in sample:
            f.write(json.dumps(row) + "\n")
    return GOLD_TEMPLATE_PATH


def score_agreement(judge_verdicts: dict[str, str]) -> dict:
    """`judge_verdicts` maps a stable item key -> the judge's label string.
    Compares against evals/gold_labels.jsonl's human_label field.
    """
    if not os.path.exists(GOLD_PATH):
        raise FileNotFoundError(
            f"{GOLD_PATH} not found - fill in {GOLD_TEMPLATE_PATH} by hand and save it as gold_labels.jsonl first"
        )
    gold = load_jsonl(GOLD_PATH)
    per_axis = {}
    for row in gold:
        axis = row["axis"]
        key = json.dumps(row["item"], sort_keys=True)
        human = row["human_label"]
        judge = judge_verdicts.get(key)
        if human is None or judge is None:
            continue
        bucket = per_axis.setdefault(axis, [0, 0])
        bucket[1] += 1
        if human == judge:
            bucket[0] += 1
    return {axis: (correct / total if total else None) for axis, (correct, total) in per_axis.items()}


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        path = write_template()
        print(f"Wrote {path} - hand-label 'human_label' for each line, save as gold_labels.jsonl")
    else:
        print("Run `python -m evals.meta_check sample` first, then hand-label and re-run with judge verdicts wired in.")
