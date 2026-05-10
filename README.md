# InfoGain-RAG: Document Information Gain-based Reranking and Filtering

[![Paper](https://img.shields.io/badge/EMNLP-2025-red)](https://arxiv.org/abs/2509.12765)

**CS5170 Advanced NLP — Final Project | Cal Poly Pomona, Spring 2026**

Replication of Wang et al. (EMNLP 2025): a Retrieval-Augmented Generation framework that scores each retrieved document's contribution using Document Information Gain (DIG), then trains a RoBERTa-large reranker to filter and rerank documents before generation.

---

## Team Members

| Name | GitHub |
|---|---|
| _(fill in before submission)_ | _(fill in)_ |
| _(fill in before submission)_ | _(fill in)_ |

---

## Paper Reference

> Zihan Wang, Zihan Liang, Zhou Shao, Yufei Ma, Huangyu Dai, Ben Chen, Lingtao Mao, Chenyi Lei, Yuqing Ding, Han Li.
> **InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering.**
> *Proceedings of EMNLP 2025*, pages 7190–7204.

---

## Pipeline Overview

The system has three stages run in sequence:

| Stage | Directory | Description |
|---|---|---|
| 1. Calculate DIG scores | `calculate_dig/` | Compute log-prob of answers with/without passages using Qwen2.5-7B via vLLM |
| 2. Train reranker | `train/` | RoBERTa-large multi-task training (RankNet + binary CE loss) |
| 3. Generate & evaluate | `generate_and_judfe/` | Rerank passages, run LLM inference, compute Exact Match |

---

## Repository Structure

```
calculate_dig/         # Stage 1 — DIG score computation (vLLM + Qwen2.5-7B)
  vllm_logits.py       # Main script: computes rag_logprob vs non_rag_logprob
  data/                # Dataset loaders (TriviaQA, NaturalQA, PopQA)

train/                 # Stage 2 — RoBERTa reranker training
  roberta_train_multi_loss_v2.py   # Primary: RankNet + CE loss (LAMBDA=0.9)
  roberta_train_ranknet_loss.py    # Ablation: RankNet only
  roberta_train_ce_loss.py         # Ablation: CE only

generate_and_judfe/    # Stage 3 — Inference and evaluation
  gen_res.py           # LLM inference (rag or no_rag mode)
  judge_res.py         # Exact Match evaluation
  rerank_passage_bert_multi.py  # Rerank passages with trained checkpoint
  tools.py             # API client wrappers (OpenAI, DeepSeek, etc.)

qa_dataset/            # Prepared evaluation datasets (JSONL)
formal_answer/         # Model outputs from evaluation runs
prepare_dataset.py     # Convert TriviaQA HuggingFace parquet → JSONL
requirements.txt
```

---

## Setup

### Requirements

- Python 3.8+
- GPU required for Stage 1 (4× GPU recommended) and Stage 2 (1× GPU)
- Stage 3 (inference + eval) runs on CPU or GPU

### Install

```bash
git clone https://github.com/keiakihito/CS5170-Sp26-Final-Info-RAG-pipeline.git
cd CS5170-Sp26-Final-Info-RAG-pipeline
pip install -r requirements.txt
```

### API Keys

Stage 3 inference requires LLM API keys. Set them as environment variables:

```bash
export OPENAI_API_KEY=your_key_here
export DEEPSEEK_API_KEY=your_key_here      # if using DeepSeek models
export DASHSCOPE_API_KEY=your_key_here     # if using Aliyun/Qwen models
export DEEPINFRA_API_KEY=your_key_here     # if using DeepInfra models
```

---

## Running the Pipeline

### Stage 1 — Prepare Dataset

Convert TriviaQA HuggingFace parquet to JSONL format:

```bash
python prepare_dataset.py \
    --trivia_parquet /path/to/validation-00000-of-00001.parquet \
    --out_dir qa_dataset/ \
    --split val \
    --limit 500
```

### Stage 2 — Generate Answers (No RAG baseline)

```bash
mkdir -p formal_answer/trivia/no_rag
python generate_and_judfe/gen_res.py \
    --qa_dataset qa_dataset/trivia_val_shuffle_500.jsonl \
    --inference_model gpt-4o-mini \
    --output_file formal_answer/trivia/no_rag/gpt-4o-mini.jsonl \
    --num_workers 1 \
    --mode no_rag
```

### Stage 3 — Generate Answers (RAG mode)

```bash
mkdir -p formal_answer/trivia/rag
python generate_and_judfe/gen_res.py \
    --qa_dataset qa_dataset/trivia_val_shuffle_500.jsonl \
    --inference_model gpt-4o-mini \
    --output_file formal_answer/trivia/rag/gpt-4o-mini.jsonl \
    --num_workers 1 \
    --mode rag
```

### Stage 4 — Evaluate (Exact Match)

```bash
python generate_and_judfe/judge_res.py \
    qa_dataset/trivia_val_shuffle_500.jsonl \
    formal_answer/trivia/no_rag/gpt-4o-mini.jsonl

python generate_and_judfe/judge_res.py \
    qa_dataset/trivia_val_shuffle_500.jsonl \
    formal_answer/trivia/rag/gpt-4o-mini.jsonl
```

---

## Results

Evaluation on TriviaQA rc.wikipedia validation split (500 questions, GPT-4o-mini).

| Model | Approach | EM / 500 | EM % |
|---|---|:---:|:---:|
| GPT-4o-mini | No RAG | 397 | 79.4% |
| GPT-4o-mini | Naive RAG (entity_pages) | 361 | 72.2% |
| GPT-4o-mini | InfoGain-RAG (reranked) | — | — |

> **Note:** Naive RAG scores lower than No RAG because `entity_pages` passages are limited (1–3 Wikipedia articles per question, not retrieved by a retrieval system). This replicates the core motivation of InfoGain-RAG: unfiltered documents hurt generation quality. The InfoGain-RAG row requires the trained reranker checkpoint (Stage 2).

Sample outputs from both runs are in `formal_answer/trivia/`.

---

## Extension Plan (Future Work)

We propose applying InfoGain-RAG to a **domain transfer** setting: evaluating whether DIG-based reranking generalizes to medical QA (e.g., MedQA or BioASQ), where retrieved documents are highly technical and noisy. The hypothesis is that DIG filtering provides even larger gains in specialized domains where naive RAG is more likely to surface irrelevant passages. This is a task transfer extension as defined in the course rubric.

---

## AI Tool Disclosure

Per the CS5170 academic integrity policy: Claude Code (Anthropic) was used for coding assistance throughout this project. All experimental results are produced by the team's own runs. Code from the original authors' repository is cited here and in comments where applicable.

---

## Citation

```bibtex
@inproceedings{wang2025infogain,
  title={InfoGain-RAG: Boosting Retrieval-Augmented Generation through Document Information Gain-based Reranking and Filtering},
  author={Wang, Zihan and Liang, Zihan and Shao, Zhou and Ma, Yufei and Dai, Huangyu and Chen, Ben and Mao, Lingtao and Lei, Chenyi and Ding, Yuqing and Li, Han},
  booktitle={Proceedings of EMNLP},
  year={2025}
}
```
