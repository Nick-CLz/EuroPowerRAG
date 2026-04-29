"""Sentiment eval — Cohen's kappa vs. hand-labeled gold set.

P1.3 — gates Phase 1 shipping. Threshold: κ ≥ 0.5.

Run:  python -m src.evaluation.eval_sentiment
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from src.sentiment.scorer import PROMPT_VERSION, discretize_score, score_article

GOLD_PATH = Path("data/eval/sentiment_gold.jsonl")
EVAL_LOG = Path("data/eval/eval_log.jsonl")
THRESHOLD = 0.5


def cohen_kappa(y_true: list[int], y_pred: list[int]) -> float:
    """Cohen's kappa with linear weights for ordinal classes {-1, 0, +1}."""
    try:
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y_true, y_pred, weights="linear"))
    except ImportError:
        # Manual computation if scikit-learn unavailable
        from collections import Counter

        n = len(y_true)
        if n == 0:
            return 0.0
        agree = sum(1 for a, b in zip(y_true, y_pred) if a == b)
        po = agree / n
        true_counts = Counter(y_true)
        pred_counts = Counter(y_pred)
        labels = set(y_true) | set(y_pred)
        pe = sum((true_counts[l] / n) * (pred_counts[l] / n) for l in labels)
        return (po - pe) / (1 - pe) if pe < 1 else 1.0


def run_eval() -> dict:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(
            f"{GOLD_PATH} not found.\n"
            f"Phase 1.1 — hand-label 30 articles into this file before running."
        )

    with open(GOLD_PATH) as f:
        gold = [json.loads(line) for line in f if line.strip()]

    if len(gold) < 10:
        raise ValueError(f"Need ≥ 10 gold examples, got {len(gold)}.")

    y_true: list[int] = []
    y_pred: list[int] = []
    per_item = []

    for i, item in enumerate(gold):
        try:
            pred = score_article(item["title"], item["summary"])
            label_pred = discretize_score(pred.score)
            y_true.append(int(item["label"]))
            y_pred.append(label_pred)
            per_item.append(
                {
                    "id": item.get("id", f"gold_{i}"),
                    "true": item["label"],
                    "pred_score": pred.score,
                    "pred_label": label_pred,
                    "agree": int(item["label"]) == label_pred,
                }
            )
        except Exception as e:
            print(f"  [{i}] Error: {e}")

    kappa = cohen_kappa(y_true, y_pred)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / max(len(y_true), 1)

    summary = {
        "n_examples": len(y_true),
        "kappa": round(kappa, 3),
        "accuracy": round(accuracy, 3),
        "passed": kappa >= THRESHOLD,
        "threshold": THRESHOLD,
        "prompt_version": PROMPT_VERSION,
    }

    print("\n" + "=" * 60)
    print(f"Sentiment Eval — prompt {PROMPT_VERSION}")
    print(f"  N:        {summary['n_examples']}")
    print(f"  κ:        {summary['kappa']}  (threshold: {THRESHOLD})")
    print(f"  Accuracy: {summary['accuracy']}")
    print(f"  Passed:   {'✅ YES' if summary['passed'] else '❌ NO'}")
    print("=" * 60)

    # Append to eval log
    EVAL_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "eval_name": "sentiment_kappa",
        "version": PROMPT_VERSION,
        "metric": summary["kappa"],
        "threshold": THRESHOLD,
        "passed": summary["passed"],
        "n_examples": summary["n_examples"],
    }
    with open(EVAL_LOG, "a") as f:
        f.write(json.dumps(log_row) + "\n")

    return summary


if __name__ == "__main__":
    run_eval()
