# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Reference implementation of **InfoGain-RAG** (EMNLP 2025), a framework that enhances RAG systems via Document Information Gain (DIG) — a metric quantifying each document's contribution to correct answer generation. The pipeline trains a RoBERTa-based reranker using multi-task learning (RankNet + binary classification loss) and evaluates it across NaturalQA, TriviaQA, and PopQA benchmarks.

## Pipeline

The system has three stages run in sequence:

**1. Calculate DIG scores** (`calculate_dig/`)  
`vllm_logits.py` loads Qwen2.5-7B-Instruct via vLLM (requires 4 GPUs at 50% utilization), computes log-probabilities for answers both with and without retrieved passages (`rag_logprob` vs `non_rag_logprob`), and writes results as Parquet to `datasets/query_doc_pair/total_qwen/`.

Input data must first be converted from JSON to Parquet format using `calculate_dig/data/query_doc_pair.py`.

**2. Train the reranker** (`train/`)  
Three training variants:
- `roberta_train_multi_loss_v2.py` — **primary**: RankNet loss + binary CE loss, controlled by `LAMBDA=0.9`
- `roberta_train_ranknet_loss.py` — RankNet loss only
- `roberta_train_ce_loss.py` — CE loss only

All variants use `roberta-large` with a shared `pre_classifier → rank_classifier` head. Training input is a JSONL file with `query` and `documents` fields (first document is the positive). Checkpoints are saved to `multitask_checkpoints/weight_0.9/`.

**3. Generate answers and evaluate** (`generate_and_judfe/`)

Rerank passages with the trained model:
```bash
python generate_and_judfe/rerank_passage_bert_multi.py \
  --model_path <checkpoint.pt> \
  --input_file <qa_dataset.jsonl> \
  --output_file <reranked.jsonl>
```

Run inference (RAG or no-RAG) against an LLM:
```bash
python generate_and_judfe/gen_res.py \
  --qa_dataset <input.jsonl> \
  --inference_model <model_name> \
  --output_file <output.jsonl> \
  --num_workers 4 \
  --mode rag   # or no_rag
```

Evaluate accuracy:
```bash
python generate_and_judfe/judge_res.py <test_path.jsonl> <results.jsonl>
```

Run the full benchmark for a model (all datasets × all conditions):
```bash
cd generate_and_judfe && bash run_single_model.sh
```

## Data formats

- QA dataset JSONL: `{"question": str, "answers": [str], "top_passages": [{"id", "title", "text"}]}`
- Training JSONL: `{"query": str, "documents": [str]}` — first document is the positive example
- DIG computation uses Parquet with columns: `query`, `answers`, `passage_text`, `passage_title`

## API keys

`generate_and_judfe/tools.py` contains placeholder API keys (`xxxxxxxxxxxxxxxxxxxxx`) for multiple providers (Aliyun/DashScope, OpenAI proxy, Llama API, AIML API, DeepSeek, DeepInfra). Replace these before running. Model routing is determined by model name string matching in `gen_res.py`.

## Hardware requirements

- DIG calculation (`vllm_logits.py`): 4 GPUs, configured via `tensor_parallel_size=4`
- Reranker training: single GPU (`cuda:1` hardcoded in training scripts)
- Reranker inference: single GPU (`cuda:1` hardcoded in `rerank_passage_bert_multi.py`)
