"""
Convert qa_dataset JSONL → reranker training JSONL.

Each output record:
    {"query": str, "documents": [pos_doc, neg_doc, neg_doc, ...]}

The first passage in top_passages is treated as the positive document.
Remaining passages are negatives. Records with fewer than 2 passages are
skipped because the reranker needs at least one positive and one negative.

Usage:
    python prepare_training_data.py \
        --input  qa_dataset/trivia_val_shuffle_500.jsonl \
        --output qa_dataset/train_query_documents.jsonl
"""

import argparse
import json
import os


def passage_to_text(p: dict) -> str:
    return f"{p['title']} {p['text']}"


def convert(input_path: str, output_path: str, min_docs: int = 2):
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    written = 0
    skipped = 0
    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            record = json.loads(line)
            passages = record.get("top_passages", [])

            if len(passages) < min_docs:
                skipped += 1
                continue

            documents = [passage_to_text(p) for p in passages]

            fout.write(json.dumps({
                "query": record["question"],
                "documents": documents,
            }, ensure_ascii=False) + "\n")
            written += 1

    print(f"Written : {written}")
    print(f"Skipped : {skipped} (fewer than {min_docs} passages)")
    print(f"Output  : {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="qa_dataset/trivia_val_shuffle_500.jsonl")
    parser.add_argument("--output", default="qa_dataset/train_query_documents.jsonl")
    parser.add_argument("--min_docs", type=int, default=2,
                        help="Minimum number of passages required (1 pos + at least 1 neg)")
    args = parser.parse_args()
    convert(args.input, args.output, args.min_docs)


if __name__ == "__main__":
    main()
