"""Judging the judge.

The spec asks us to assess the quality of the judge. The obvious approach -
hand-label some outputs and measure agreement - has a problem: at this
scale the labels would be produced by the same kind of system doing the
judging, which is circular, and a 15-item raw agreement figure overstates
reliability because it doesn't correct for chance.

So the judge is scored against ground truth the dataset already carries.
MedHallu ships, for each question, a known-correct answer AND a known
hallucinated one. Feeding both to the judge gives items whose right verdict
is known independently of any model, which supports real classification
metrics rather than a single agreement number:

  - accuracy, precision, recall, F1 on detecting hallucination
  - Cohen's kappa, which corrects for agreement that chance alone explains
  - the confusion matrix, because which way it fails matters: a judge that
    misses hallucinations flatters the agents, one that over-flags
    punishes them, and those are not the same defect

The safety axis gets a second, independent check: the deterministic regex
classifier and the LLM judge label the same responses, and their
disagreement rate is reported. Two mechanisms agreeing is weak evidence
they are right; disagreeing is strong evidence one is wrong.
"""
from __future__ import annotations

import json
import logging
import os
import time

from evals.judge import HALLUCINATION_PROMPT, Judge

logger = logging.getLogger("wellness.meta")

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "datasets")
RESULTS_DIR = os.path.join(os.path.dirname(HERE), "results")


def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cohens_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Agreement corrected for what chance alone would produce.

    Raw agreement on a balanced two-class problem starts at 50% for a coin
    flip, so quoting it uncorrected makes a useless judge look adequate.
    kappa 0 = no better than chance, 1 = perfect.
    """
    n = tp + fp + fn + tn
    if n == 0:
        return None
    observed = (tp + tn) / n
    p_yes = ((tp + fp) / n) * ((tp + fn) / n)
    p_no = ((fn + tn) / n) * ((fp + tn) / n)
    expected = p_yes + p_no
    if expected >= 1.0:
        return None
    return round((observed - expected) / (1 - expected), 4)


def evaluate_judge(progress=None) -> dict:
    """Asks the judge to label known-correct and known-hallucinated answers.

    'positive' = the judge says hallucinated.
    """
    items = load_jsonl(os.path.join(DATA_DIR, "hallucination.jsonl"))
    judge = Judge()

    cases = []
    for item in items:
        reference = (item.get("context") or "").strip()
        reference = f"{reference}\n\nKnown correct answer: {item['ground_truth']}".strip()
        # the known-good answer: the judge should say "supported"
        cases.append({"item": item, "answer": item["ground_truth"],
                      "truth": "supported", "reference": reference})
        # the known-bad answer: the judge should say "hallucinated"
        if item.get("hallucinated_answer"):
            cases.append({"item": item, "answer": item["hallucinated_answer"],
                          "truth": "hallucinated", "reference": reference})

    tp = fp = fn = tn = 0
    details = []
    for i, case in enumerate(cases):
        try:
            v = judge.ask(HALLUCINATION_PROMPT.format(
                reference=case["reference"],
                question=case["item"]["question"],
                answer=case["answer"]))
            predicted = v.get("verdict")
            reason = v.get("reason")
        except Exception as e:  # noqa: BLE001
            details.append({"truth": case["truth"], "predicted": None,
                            "error": f"{type(e).__name__}: {str(e)[:120]}"})
            continue

        truth = case["truth"]
        if truth == "hallucinated" and predicted == "hallucinated":
            tp += 1
        elif truth == "supported" and predicted == "hallucinated":
            fp += 1
        elif truth == "hallucinated" and predicted == "supported":
            fn += 1
        elif truth == "supported" and predicted == "supported":
            tn += 1

        details.append({"question": case["item"]["question"][:120],
                        "truth": truth, "predicted": predicted,
                        "correct": truth == predicted, "reason": reason})
        if progress:
            progress(i + 1, len(cases), "judge-meta")

    n = tp + fp + fn + tn
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall) else None

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_model": judge.model,
        "method": "dataset ground truth (MedHallu correct vs hallucinated answer pairs)",
        "n_cases": n,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "cohens_kappa": cohens_kappa(tp, fp, fn, tn),
        "confusion_matrix": {
            "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
        },
        "judge_calls": judge.calls,
        "errors": len([d for d in details if d.get("error")]),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "judge_quality.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": report, "cases": details}, f, indent=2)
    return report


def safety_cross_check() -> dict:
    """How often the LLM judge and the deterministic classifier disagree on
    the same safety responses. High disagreement means at least one is
    unreliable and the safety numbers need reading with care."""
    raw_path = os.path.join(RESULTS_DIR, "raw.jsonl")
    if not os.path.exists(raw_path):
        return {"error": "results/raw.jsonl not found - run the evals first"}

    rows = [r for r in load_jsonl(raw_path)
            if r.get("axis") == "safety" and r.get("verdict") and r.get("rule_verdict")]
    if not rows:
        return {"error": "no scored safety rows"}

    agree = 0
    for r in rows:
        rule_says_refused = r["rule_verdict"] in {"appropriate_refusal", "over_refusal"}
        judge_says_refused = r["verdict"] == "refused"
        if rule_says_refused == judge_says_refused:
            agree += 1

    out = {
        "n": len(rows),
        "agreement": round(agree / len(rows), 4),
        "disagreements": len(rows) - agree,
        "note": "rule-based regex classifier vs LLM judge on the same responses",
    }
    with open(os.path.join(RESULTS_DIR, "safety_cross_check.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    report = evaluate_judge(progress=lambda d, t, label: print(f"  [{d}/{t}] {label}"))
    print(json.dumps(report, indent=2))
    print(json.dumps(safety_cross_check(), indent=2))


if __name__ == "__main__":
    main()
