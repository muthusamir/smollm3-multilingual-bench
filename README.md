# SmolLM3 Benchmark

Deployable Python script for benchmarking **SmolLM3** on multilingual QA datasets (XQuAD, MLQA, GSM8K).  
Evaluates models with **Exact Match (EM)**, **F1**, inference time, memory usage, and a custom **Resource Efficiency Index (REI)**.

---

## ✨ Features
- Benchmark Hugging Face models on multilingual QA datasets
- Supports reasoning modes: `think` vs `no_think`
- Tracks EM, F1, inference time, memory usage
- Computes REI (Resource Efficiency Index)
- Saves results to JSON/CSV for easy analysis
- Works on both CPU and GPU

---

## 📦 Requirements
Install dependencies with pip:

```bash
pip install transformers datasets torch psutil numpy tqdm
