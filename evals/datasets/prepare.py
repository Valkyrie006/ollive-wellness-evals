"""Pulls ~15-item samples from 3 ready-made Hugging Face datasets - one per
eval axis - so no test items have to be hand-written (plan.md Section 5).

NOTE: HF dataset schemas can change. If a `transform` lambda below raises a
KeyError, open the dataset's card on huggingface.co and fix the field name.
"""
import json
import random

random.seed(0)


def sample_and_save(dataset_id: str, split: str, n: int, out_path: str, transform):
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split)
    idx = random.sample(range(len(ds)), min(n, len(ds)))
    with open(out_path, "w", encoding="utf-8") as f:
        for i in idx:
            f.write(json.dumps(transform(ds[i])) + "\n")
    return len(idx)


def main():
    n = sample_and_save(
        "UTAustin-AIHealth/MedHallu", "pqa_labeled", 15, "evals/datasets/hallucination.jsonl",
        lambda r: {
            "question": r["Question"],
            "context": r["Knowledge"],
            "ground_truth": r["Ground Truth"],
            "hallucinated_answer": r.get("Hallucinated Answer"),
        },
    )
    print(f"hallucination.jsonl: {n} items")

    n = sample_and_save(
        "walledai/BBQ", "test", 15, "evals/datasets/bias.jsonl",
        lambda r: {"question": r["question"], "category": r.get("category")},
    )
    print(f"bias.jsonl: {n} items")

    n = sample_and_save(
        "walledai/JailbreakBench", "train", 15, "evals/datasets/safety.jsonl",
        lambda r: {"prompt": r.get("prompt") or r.get("Behavior") or r.get("behavior")},
    )
    print(f"safety.jsonl: {n} items")


if __name__ == "__main__":
    main()
