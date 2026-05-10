"""
Generate a bar chart comparing No RAG, Naive RAG, and InfoGain-RAG EM scores.

Usage:
    # With all three results:
    python visualize_results.py \
        --no_rag   outputs/trivia/no_rag/gpt-4o-mini.jsonl \
        --rag      outputs/trivia/rag/gpt-4o-mini.jsonl \
        --reranked outputs/trivia/reranked/gpt-4o-mini.jsonl \
        --dataset  qa_dataset/trivia_val_shuffle_500.jsonl \
        --out      outputs/trivia_em_comparison.png

    # Without reranked (just no_rag vs rag):
    python visualize_results.py \
        --no_rag  outputs/trivia/no_rag/gpt-4o-mini.jsonl \
        --rag     outputs/trivia/rag/gpt-4o-mini.jsonl \
        --dataset qa_dataset/trivia_val_shuffle_500.jsonl \
        --out     outputs/trivia_em_comparison.png
"""

import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import seaborn as sns


def regularization(text: str) -> str:
    """Same normalization as judge_res.py."""
    result = re.sub(r'[.,]', '', text)
    result = re.sub(r'[\"\']', '', result)
    result = result.split('\n')[0]
    result = result.lower()
    result = result.split(" was a ", 1)[1].strip() if " was a " in result else result
    result = result.split(" was an ", 1)[1].strip() if " was an " in result else result
    result = result.split(" was ", 1)[1].strip() if " was " in result else result
    result = result.split("the answer is", 1)[1].strip() if "the answer is" in result else result
    return result.lower().strip()


def has_answer_loose(output: str, answer_list: list) -> bool:
    for answer in answer_list:
        if answer in output or output in answer:
            return True
    return False


def load_dataset(path: str) -> list:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_em(pred_path: str, dataset: list) -> float:
    preds = []
    with open(pred_path, "r", encoding="utf-8") as f:
        for line in f:
            preds.append(json.loads(line))

    correct = 0
    for i in range(min(500, len(dataset), len(preds))):
        try:
            pred = regularization(preds[i].get("answer", ""))
            gts = [a.lower().strip() for a in dataset[i]["answers"]]
            if has_answer_loose(pred, gts):
                correct += 1
        except Exception:
            pass
    return 100.0 * correct / 500


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_rag",   required=True)
    parser.add_argument("--rag",      required=True)
    parser.add_argument("--reranked", default=None, help="InfoGain-RAG output (optional)")
    parser.add_argument("--dataset",  required=True, help="Ground-truth QA dataset JSONL")
    parser.add_argument("--out",      default="outputs/trivia_em_comparison.png")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)

    labels = ["No RAG", "Naive RAG"]
    scores = [compute_em(args.no_rag, dataset), compute_em(args.rag, dataset)]

    if args.reranked and os.path.exists(args.reranked):
        labels.append("InfoGain-RAG")
        scores.append(compute_em(args.reranked, dataset))

    print("EM scores:")
    for label, score in zip(labels, scores):
        print(f"  {label}: {score:.1f}%")

    sns.set_theme(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("muted", len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, scores, color=palette, width=0.5, edgecolor="white")

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{score:.1f}%",
            ha="center", va="bottom", fontweight="bold"
        )

    ax.set_ylabel("Exact Match (%)")
    ax.set_title("TriviaQA — GPT-4o-mini: No RAG vs Naive RAG vs InfoGain-RAG")
    ax.set_ylim(0, min(100, max(scores) + 10))

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Chart saved → {args.out}")


if __name__ == "__main__":
    main()
