#!/usr/bin/env python3
# benchmark_smollm3.py
# Deployable Python script for benchmarking SmolLM3 on multilingual QA datasets
# Requirements: transformers, datasets, torch, psutil, numpy, tqdm
# Optional: pandas (for CSV export)
# Example:
#   python benchmark_smollm3.py --model HuggingFaceTB/SmolLM3-3B-Instruct --dataset xquad --dataset_config xquad.en --mode think --device cuda --limit 100

import argparse
import time
import json
import os
import math
import numpy as np
import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, set_seed
from datasets import load_dataset
from tqdm import tqdm
import logging

try:
    import pandas as pd
    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------
# Text normalization & metrics
# ----------------------------
def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    import string, re
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0
    common = set(prediction_tokens) & set(ground_truth_tokens)
    if len(common) == 0:
        return 0.0
    precision = len(common) / len(prediction_tokens)
    recall = len(common) / len(ground_truth_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_rei(f1_fraction: float, inference_time: float, memory_gb: float) -> float:
    """Resource Efficiency Index (toy metric): higher is better."""
    if inference_time <= 0 or memory_gb <= 0 or f1_fraction <= 0:
        return 0.0
    # Avoid log10 of < 1 by clamping product minimally
    product = max(inference_time * memory_gb, 1e-9)
    return f1_fraction / math.log10(product + 1e-12)


# ----------------------------
# Memory profiling helpers
# ----------------------------
def current_cpu_memory_gb() -> float:
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 ** 3)  # GB


def get_peak_memory_gb() -> float:
    """Return peak memory in GB (CUDA if available, else current RSS)."""
    if torch.cuda.is_available():
        peak_bytes = torch.cuda.max_memory_allocated()
        return peak_bytes / (1024 ** 3)
    return current_cpu_memory_gb()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ----------------------------
# Inference
# ----------------------------
def build_prompt(question: str, context: str, mode: str = "no_think") -> str:
    if mode == "think":
        return f"Think step by step before answering.\n\nContext: {context}\nQuestion: {question}\nAnswer:"
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def extract_answer_from_output(output_text: str) -> str:
    # Heuristic: take text after "Answer:"
    if "Answer:" in output_text:
        return output_text.split("Answer:", 1)[-1].strip()
    return output_text.strip()


def run_inference(pipe, question: str, context: str, mode: str, max_new_tokens: int, temperature: float, top_p: float):
    prompt = build_prompt(question, context, mode)
    reset_peak_memory()
    start_time = time.time()
    with torch.no_grad():
        outputs = pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature is not None and temperature > 0),
            temperature=(temperature if temperature is not None else 0.0),
            top_p=(top_p if top_p is not None else 1.0),
            truncation=True,
        )
    inference_time = time.time() - start_time
    peak_memory = get_peak_memory_gb()
    # Pipeline returns list of dicts with "generated_text"
    output_text = outputs[0]["generated_text"]
    answer = extract_answer_from_output(output_text)
    return answer, inference_time, peak_memory, output_text


# ----------------------------
# Dataset utilities
# ----------------------------
def prepare_dataset(dataset_name: str, dataset_config: str = None, split: str = None, limit: int = None, seed: int = 42):
    """Load and standardize datasets into a list of dicts with fields: question, context (optional), ground_truth."""
    # Default splits for common datasets
    default_split = split or ("validation" if dataset_name in ["xquad", "mlqa"] else "test")
    if dataset_config:
        ds = load_dataset(dataset_name, dataset_config, split=default_split)
    else:
        ds = load_dataset(dataset_name, split=default_split)

    # Shuffle and limit
    ds = ds.shuffle(seed=seed)
    if limit is not None and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))

    examples = []
    for ex in ds:
        if dataset_name in ["xquad", "mlqa"]:
            # Expect extractive QA format
            question = ex.get("question", "")
            context = ex.get("context", "")
            answers = ex.get("answers", {})
            gt_list = answers.get("text", []) if isinstance(answers, dict) else []
            ground_truth = gt_list[0] if gt_list else ""
        elif dataset_name == "gsm8k":
            # GSM8K: question + final answer in "answer" (may include reasoning, take last line numeric)
            question = ex.get("question", "")
            context = ""  # no context for gsm8k
            raw_answer = ex.get("answer", "")
            # Heuristic: extract final answer after '#### ' as in GSM8K format
            if "####" in raw_answer:
                ground_truth = raw_answer.split("####")[-1].strip()
            else:
                ground_truth = raw_answer.strip()
        else:
            # Generic fallback: try common fields
            question = ex.get("question", "")
            context = ex.get("context", "")
            ground_truth = ex.get("answer", "")
            if isinstance(ground_truth, dict):
                gt_list = ground_truth.get("text", [])
                ground_truth = gt_list[0] if gt_list else ""

        examples.append({
            "question": question,
            "context": context,
            "ground_truth": ground_truth
        })
    return examples


# ----------------------------
# Benchmark routine
# ----------------------------
def benchmark(model_name: str,
             dataset_name: str,
             mode: str,
             device: str,
             limit: int,
             dataset_config: str = None,
             split: str = None,
             max_new_tokens: int = 64,
             temperature: float = 0.0,
             top_p: float = 1.0,
             seed: int = 42,
             save_dir: str = "outputs",
             save_predictions: bool = True):
    # Reproducibility
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Load tokenizer and model
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure pad token is set for generation pipelines
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    # Device setup
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        pipe_device = 0
    else:
        model = model.to("cpu")
        pipe_device = -1

    # Build generation pipeline
    text_gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=pipe_device
    )

    # Load dataset
    logger.info(f"Loading dataset: {dataset_name} (config={dataset_config}, split={split})")
    examples = prepare_dataset(dataset_name, dataset_config=dataset_config, split=split, limit=limit, seed=seed)
    if len(examples) == 0:
        raise ValueError("No examples loaded from dataset; check dataset name/config/split.")

    # Metrics accumulators
    em_scores, f1_scores, times, memories = [], [], [], []
    records = []

    # Iterate and evaluate
    for ex in tqdm(examples, desc="Benchmarking"):
        q = ex["question"]
        ctx = ex["context"]
        gt = ex["ground_truth"]

        pred, inf_time, mem_gb, raw_out = run_inference(
            text_gen,
            question=q,
            context=ctx,
            mode=mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )

        em = exact_match_score(pred, gt)
        f1 = f1_score(pred, gt)

        em_scores.append(em)
        f1_scores.append(f1)
        times.append(inf_time)
        memories.append(mem_gb)

        records.append({
            "question": q,
            "context": ctx,
            "ground_truth": gt,
            "prediction": pred,
            "raw_output": raw_out,
            "inference_time_sec": inf_time,
            "peak_memory_gb": mem_gb,
            "em": em,
            "f1": f1
        })

    # Aggregate metrics
    avg_em = float(np.mean(em_scores)) * 100.0
    avg_f1 = float(np.mean(f1_scores)) * 100.0
    avg_time = float(np.mean(times))
    avg_mem = float(np.mean(memories))
    rei = compute_rei((avg_f1 / 100.0), avg_time, avg_mem)

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "split": split,
        "mode": mode,
        "limit": limit,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "metrics": {
            "avg_em_percent": round(avg_em, 4),
            "avg_f1_percent": round(avg_f1, 4),
            "avg_inference_time_sec": round(avg_time, 6),
            "avg_peak_memory_gb": round(avg_mem, 6),
            "rei": round(rei, 6)
        }
    }

    # Logging summary
    logger.info(f"Avg EM: {summary['metrics']['avg_em_percent']:.2f}%")
    logger.info(f"Avg F1: {summary['metrics']['avg_f1_percent']:.2f}%")
    logger.info(f"Avg Inference Time: {summary['metrics']['avg_inference_time_sec']:.4f}s")
    logger.info(f"Avg Peak Memory: {summary['metrics']['avg_peak_memory_gb']:.4f} GB")
    logger.info(f"REI: {summary['metrics']['rei']:.4f}")

    # Save outputs
    os.makedirs(save_dir, exist_ok=True)
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if save_predictions:
        preds_json_path = os.path.join(save_dir, "predictions.jsonl")
        with open(preds_json_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if HAS_PANDAS:
            preds_csv_path = os.path.join(save_dir, "predictions.csv")
            pd.DataFrame(records).to_csv(preds_csv_path, index=False)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SmolLM3 on QA datasets")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM3-3B-Instruct", help="HF model name")
    parser.add_argument("--dataset", default="xquad", choices=["xquad", "mlqa", "gsm8k"], help="Dataset name")
    parser.add_argument("--dataset_config", default=None, help="Dataset config/language (e.g., xquad.en, mlqa.en.en)")
    parser.add_argument("--split", default=None, help="Dataset split (default: validation for xquad/mlqa, test for gsm8k)")
    parser.add_argument("--mode", default="no_think", choices=["think", "no_think"], help="Reasoning mode")

    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"],
                        help="Device: cuda or cpu")

    parser.add_argument("--limit", type=int, default=100, help="Number of examples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for deterministic)")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p for nucleus sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--save_dir", default="outputs", help="Directory to save results")
    parser.add_argument("--no_save_predictions", action="store_true", help="If set, do not save per-example predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    benchmark(
        model_name=args.model,
        dataset_name=args.dataset,
        mode=args.mode,
        device=args.device,
        limit=args.limit,
        dataset_config=args.dataset_config,
        split=args.split,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        save_dir=args.save_dir,
        save_predictions=(not args.no_save_predictions)
    )


if __name__ == "__main__":
    main()
