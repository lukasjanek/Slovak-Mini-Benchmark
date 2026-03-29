# Slovak NLP Benchmark Mini

An open benchmark for evaluating language models on Slovak NLP tasks.
Designed to run on Google Colab with GPU support.

---

## 📋 Tasks

| Task | Dataset | Split | Metric |
|---|---|---|---|
| ❓ Question Answering | [lukasjanek/skquad](https://huggingface.co/datasets/lukasjanek/skquad) | validation | Token F1 |
| 💬 Sentiment Classification | [lukasjanek/senti-sk](https://huggingface.co/datasets/lukasjanek/senti-sk) | test | F1 Weighted |
| 🔤 Fill Mask | [lukasjanek/slovaksum](https://huggingface.co/datasets/lukasjanek/slovaksum) | validation | Top-1 Accuracy |
| ✍️ Text Generation | [lukasjanek/slovaksum](https://huggingface.co/datasets/lukasjanek/slovaksum) | test | Perplexity / BPC |

---

## 🚀 Quick Start
Create a hugging face token, give it all necessary perminssions.

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/slovak-benchmark-mini
cd slovak-benchmark-mini

# Install dependencies
pip install -r requirements.txt

# Run a single task
python run_benchmark.py --model YOUR_MODEL --task fill_mask

# Run all applicable tasks for a model (auto-detected)
python run_benchmark.py --model YOUR_MODEL --mode full

# Interactive mode
python run_benchmark.py -i
```

## 🚀 Quick Start for Google Colab
Import the hugging face key as secret, name it HF_TOKEN and add value, check the Notebook access otherwise will be prompted on run.

```bash

!unzip slovak-benchmark-mini.zip -d /content/
%cd /content/slovak-benchmark-mini
!python colab_setup.py

from huggingface_hub import login, whoami
login()
print(whoami())

!python run_benchmark.py --help

```

---

## 🛠️ Usage

```bash
# Specify sample count
python run_benchmark.py --model YOUR_MODEL --task qa --samples 500

# Override auto-detected model type
python run_benchmark.py --model YOUR_MODEL --task qa --model-type extractive

# View leaderboard
python leaderboard.py --task qa
python leaderboard.py --task fill_mask

# Export leaderboard as CSV
python leaderboard.py --task classification --csv

# Show dataset limits
python run_benchmark.py --limits
```

Tasks auto-select based on model architecture:
- `masked` (BERT/RoBERTa) → fill_mask, classification
- `extractive` → qa
- `generative_causal` (GPT) → generation
- `generative_seq2seq` (T5) → qa, generation

---

## 📁 File Structure

```
slovak-benchmark-mini/
├── config.py           — dataset limits, splits, random seeds
├── evaluation.py       — type detection, benchmark orchestration, tokenizer checks
├── leaderboard.py      — leaderboard display and CSV export
├── run_benchmark.py    — CLI entry point
├── colab_setup.py      — Colab helper utilities
├── requirements.txt
├── tasks/
│   ├── qa.py           — extractive + generative QA
│   ├── classification.py — sentiment with auto label alignment
│   ├── fill_mask.py    — masked token prediction
│   └── generation.py  — perplexity and BPC evaluation
└── results/            — JSON output files, one per run
```

---

## Metrics

**Overall score** — weighted average of QA (Token F1) and Classification (F1 Weighted) only. Fill-mask and generation are reported separately since perplexity cannot be fairly averaged with F1.

**BPC (Bits per Character)** — primary generation metric. Normalises perplexity by character count so models with different tokenisers are directly comparable.

| BPC | Rating |
|---|---|
| < 0.5 | ✅ Excellent |
| 0.5 – 1.0 | 🟡 Good |
| 1.0 – 2.0 | 🟠 Fair |
| > 2.0 | 🔴 Poor |
 
| Perplexity | Rating |
|---|---|
| < 20 | ✅ Excellent |
| 20 – 50 | 🟡 Good |
| 50 – 100 | 🟠 Fair |
| > 100 | 🔴 Poor |

**Classification label alignment** — models with different label schemes are automatically aligned to the dataset:
- Fine-grained labels (e.g. "Very Negative") are merged to coarser classes
- 2-class models are evaluated on positive/negative only (neutral samples filtered)

---

## Datasets

All datasets are hosted on HuggingFace:

| Dataset | Task | Samples |
|---|---|---|
| `lukasjanek/skquad` | ❓ QA | 9,583 |
| `lukasjanek/senti-sk` | 💬 Classification | 3,400 |
| `lukasjanek/slovaksum` (validation) | 🔤 Fill Mask | 20,663 |
| `lukasjanek/slovaksum` (test) | ✍️ Generation | 20,662 |

---

## ⚙️ Requirements

```
transformers>=4.40.0
datasets>=2.18.0
torch>=2.2.0
bitsandbytes>=0.43.0
bert-score>=0.3.13
scikit-learn>=1.4.0
huggingface-hub>=0.20.0
tqdm
```

GPU with at least 16 GB VRAM recommended for models 7B+. Smaller models run on any GPU.

---

## 📄 License

MIT. Datasets and models used are subject to their own licenses — see individual HuggingFace cards.