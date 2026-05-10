"""
Convert local TriviaQA HuggingFace parquet cache → qa_dataset JSONL
format expected by generate_and_judfe/gen_res.py.

Usage:
    python prepare_dataset.py \
        --trivia_parquet /path/to/validation-00000-of-00001.parquet \
        --out_dir qa_dataset/ \
        --split val \
        --limit 500
"""

import argparse
import json
import os

import numpy as np
import pandas as pd


def _safe_list(val):
    """Convert numpy array / list / scalar to a plain Python list."""
    if val is None:
        return []
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, list):
        return val
    return [val]


def convert_triviaqa_parquet(parquet_path: str, out_path: str, limit: int = 500):
    """
    TriviaQA rc.wikipedia parquet columns:
        question, question_id, answer (dict with 'aliases', 'value'),
        entity_pages (dict with 'title', 'wiki_context', 'filename')

    Output JSONL (one line per question):
        {"question": str, "answers": [str, ...],
         "top_passages": [{"id": str, "title": str, "text": str}, ...]}
    """
    df = pd.read_parquet(parquet_path)
    df = df.head(limit)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            question = row["question"]

            # --- answers ---
            ans_field = row["answer"]
            if isinstance(ans_field, dict):
                aliases = _safe_list(ans_field.get("aliases", []))
                value   = ans_field.get("value", "")
                answers = list({value} | set(aliases)) if value else aliases
            else:
                answers = [str(ans_field)]
            answers = [a for a in answers if a]

            # --- passages from entity_pages ---
            ep = row.get("entity_pages", {})
            titles   = _safe_list(ep.get("title", []))
            contexts = _safe_list(ep.get("wiki_context", []))
            filenames = _safe_list(ep.get("filename", []))

            passages = []
            for i, (title, ctx) in enumerate(zip(titles, contexts)):
                if ctx:
                    passages.append({
                        "id":    filenames[i] if i < len(filenames) else f"p{i}",
                        "title": title,
                        "text":  str(ctx)[:2000],   # truncate very long articles
                    })

            # skip questions with no passages
            if not passages:
                continue

            record = {
                "question":     question,
                "answers":      answers,
                "top_passages": passages,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trivia_parquet",
        default=(
            "../Final/dataset/TriviaQA/datasets--mandarjoshi--trivia_qa"
            "/snapshots/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f"
            "/rc.wikipedia/validation-00000-of-00001.parquet"
        ),
        help="Path to TriviaQA rc.wikipedia parquet file",
    )
    parser.add_argument("--out_dir", default="qa_dataset")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    out_path = os.path.join(args.out_dir, f"trivia_{args.split}_shuffle_{args.limit}.jsonl")
    convert_triviaqa_parquet(args.trivia_parquet, out_path, limit=args.limit)


if __name__ == "__main__":
    main()
