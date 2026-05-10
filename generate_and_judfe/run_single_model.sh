#!/bin/bash

model="qwen2.5-72b-instruct"

output_file="outputs/trivia/no_rag/${model}.jsonl"
mkdir -p "outputs/trivia/no_rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode no_rag

python judge_res.py \
    "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    "$output_file"

output_file="outputs/trivia/rag/${model}.jsonl"
mkdir -p "outputs/trivia/rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
    "qa_dataset/trivia_test_shuffle_2000.jsonl" \
    "$output_file"


output_file="outputs/trivia/bge-rerank/${model}.jsonl"
mkdir -p "outputs/trivia/bge-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_500_rerank_large.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/trivia_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="outputs/trivia/multi-rerank/${model}.jsonl"
mkdir -p "outputs/trivia/multi-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/trivia_test_shuffle_500_rerank_bert_multi_0.9weight_0.16.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/trivia_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="outputs/nq/no_rag/${model}.jsonl"
mkdir -p "outputs/nq/no_rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode no_rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="outputs/nq/rag/${model}.jsonl"
mkdir -p "outputs/nq/rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="outputs/nq/bge-rerank/${model}.jsonl"
mkdir -p "outputs/nq/bge-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_500_rerank_large.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="outputs/nq/multi-rerank/${model}.jsonl"
mkdir -p "outputs/nq/multi-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/nq_test_shuffle_500_rerank_bert_multi_0.9weight_0.16.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/nq_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="outputs/pop/no_rag/${model}.jsonl"
mkdir -p "outputs/pop/no_rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode no_rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"

output_file="outputs/pop/rag/${model}.jsonl"
mkdir -p "outputs/pop/rag"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_2000.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"
        
output_file="outputs/pop/bge-rerank/${model}.jsonl"
mkdir -p "outputs/pop/bge-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_500_rerank_large.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"
        
output_file="outputs/pop/multi-rerank/${model}.jsonl"
mkdir -p "outputs/pop/multi-rerank"
echo "Running generation for $output_file..."
python gen_res.py \
    --qa_dataset "qa_dataset/pop_test_shuffle_500_rerank_bert_multi_0.9weight_0.16.jsonl" \
    --inference_model "$model" \
    --output_file "$output_file" \
    --num_workers 1 \
    --mode rag

python judge_res.py \
        "qa_dataset/pop_test_shuffle_2000.jsonl" \
        "$output_file"
        
echo "All models completed!"