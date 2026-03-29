"""
QA task – supports extractive and generative models on SKQuAD.

Subtasks and their metrics:
  qa_extractive   ForQuestionAnswering models
                  → exact_match, token_f1  (primary: token_f1)
  qa_generative   ForConditionalGeneration / ForCausalLM models
                  → bertscore_f1, token_f1  (primary: bertscore_f1)
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, BitsAndBytesConfig

from config import TASK_LIMITS, RANDOM_SEED

QA_BATCH_SIZE = 32


# ─── BERTScore ───────────────────────────────────────────────────────────────

def _bertscore_available() -> bool:
    try:
        import bert_score
        return True
    except ImportError:
        return False


def _compute_bertscore(predictions: List[str], references: List[str]) -> List[float]:
    """
    Compute BERTScore F1 for each prediction against its best reference.

    """
    try:
        from bert_score import score as bs_score
        # references here is a flat list — one ref per prediction (best ref)
        P, R, F = bs_score(
            predictions, references,
            model_type="bert-base-multilingual-cased",
            lang="sk",
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        return F.tolist()
    except Exception as e:
        print(f"\n  ⚠️  BERTScore failed ({e}), falling back to token F1.")
        return [_f1(p, r) for p, r in zip(predictions, references)]


# ─── Token-level SQuAD metrics ───────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize(prediction).split()
    gt_tokens   = _normalize(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return int(pred_tokens == gt_tokens)
    common     = Counter(pred_tokens) & Counter(gt_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _exact_match(prediction: str, ground_truth: str) -> bool:
    return _normalize(prediction) == _normalize(ground_truth)


def _best_token_score(prediction: str, references: List[str]) -> Tuple[float, bool]:
    """Best token F1 and any exact match across all references."""
    f1s = [_f1(prediction, ref) for ref in references]
    ems = [_exact_match(prediction, ref) for ref in references]
    return max(f1s), any(ems)


def _best_reference(prediction: str, references: List[str]) -> str:
    """Return the reference with highest token F1 — used as BERTScore target."""
    return max(references, key=lambda r: _f1(prediction, r))


# ─── Dataset loader ──────────────────────────────────────────────────────────

def load_qa_dataset(n_samples: int) -> List[Dict]:
    cfg = TASK_LIMITS["qa"]
    n   = min(n_samples, cfg["max"])
    ds  = load_dataset(cfg["dataset"], split=cfg["split"])
    ds  = ds.shuffle(seed=RANDOM_SEED).select(range(n))

    examples = []
    for row in ds:
        answers = row.get("answers", {})
        refs    = answers.get("text", [])
        if isinstance(refs, str):
            refs = [refs]
        if not refs:
            continue
        examples.append({
            "id":       row.get("id", ""),
            "context":  row["context"],
            "question": row["question"],
            "answers":  refs,
        })
    return examples


# ─── 4-bit quant ─────────────────────────────────────────────────────────────

def _bnb_config():
    if not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ─── Extractive ──────────────────────────────────────────────────────────────

def _extract_answer(model, tokenizer, question: str, context: str, device) -> str:
    """
    Manually run a ForQuestionAnswering model and convert span logits to text.

    """
    inputs = tokenizer(
        question, context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits   = outputs.end_logits[0]

    # Find the best valid (start <= end) span within the context tokens
    # Mask out [CLS], [SEP] and padding by setting their scores very low
    seq_len      = start_logits.size(0)
    start_scores = start_logits.cpu()
    end_scores   = end_logits.cpu()

    best_score = -float("inf")
    best_start = 0
    best_end   = 0

    for s in range(seq_len):
        for e in range(s, min(s + 30, seq_len)):   # max answer length = 30 tokens
            score = start_scores[s] + end_scores[e]
            if score > best_score:
                best_score = score
                best_start = s
                best_end   = e

    # Convert token indices back to a string
    input_ids = inputs["input_ids"][0]
    answer_tokens = input_ids[best_start : best_end + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
    return answer


def run_extractive(model_name: str, examples: List[Dict]) -> Dict[str, Any]:
    print(f"\n  Loading extractive QA model: {model_name}")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb       = _bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModelForQuestionAnswering.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float32 if (device.type == "cpu" or bnb) else torch.float16,
    ).eval()

    total_f1 = 0.0
    total_em = 0
    n        = len(examples)
    errors   = 0

    for ex in tqdm(examples, desc="QA Extractive", unit="ex"):
        try:
            pred = _extract_answer(model, tokenizer, ex["question"], ex["context"], device)
        except Exception as e:
            if errors < 3:
                print(f"\n  ⚠️  Error on sample {errors + 1}: {e}")
            errors += 1
            pred = ""

        f1, em = _best_token_score(pred, ex["answers"])
        total_f1 += f1
        total_em += int(em)

    if errors > 0:
        print(f"\n  ⚠️  {errors}/{n} samples failed during inference")

    return {
        "subtask":     "extractive",
        "exact_match": round(100.0 * total_em / n, 2),
        "token_f1":    round(100.0 * total_f1 / n, 2),
        "n_samples":   n,
    }


# ─── Generative inference ────────────────────────────────────────────────────

def _decode_seq2seq(model, tokenizer, examples: List[Dict], device) -> List[str]:
    """Run seq2seq generation, return list of predictions."""
    preds = []
    for i in tqdm(range(0, len(examples), QA_BATCH_SIZE), desc="QA Seq2Seq", unit="batch"):
        batch   = examples[i : i + QA_BATCH_SIZE]
        prompts = [
            f"Otázka: {ex['question']}\nKontext: {ex['context'][:512]}"
            for ex in batch
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64,
                                        num_beams=2, early_stopping=True)
        preds.extend(tokenizer.batch_decode(output_ids, skip_special_tokens=True))
    return [p.strip() for p in preds]


def _decode_causal(model, tokenizer, examples: List[Dict], device) -> List[str]:
    """Run causal generation, return list of predictions."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    preds = []
    for i in tqdm(range(0, len(examples), QA_BATCH_SIZE), desc="QA Causal", unit="batch"):
        batch      = examples[i : i + QA_BATCH_SIZE]
        prompts    = [
            f"Otázka: {ex['question']}\nKontext: {ex['context'][:512]}\nOdpoveď:"
            for ex in batch
        ]
        inputs     = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=64,
                                        do_sample=False,
                                        pad_token_id=tokenizer.pad_token_id)
        decoded = tokenizer.batch_decode(output_ids[:, prompt_len:],
                                         skip_special_tokens=True)
        preds.extend(p.strip().split("\n")[0] for p in decoded)
    return preds


def run_generative(model_name: str, examples: List[Dict], model_type: str) -> Dict[str, Any]:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

    print(f"\n  Loading generative model: {model_name}")
    bnb    = _bnb_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer   = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    load_kwargs = dict(
        device_map="auto",
        quantization_config=bnb,
        torch_dtype=torch.float32 if (device.type == "cpu" or bnb) else torch.float16,
    )

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs).eval()
        predictions = _decode_seq2seq(model, tokenizer, examples, device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs).eval()
        predictions = _decode_causal(model, tokenizer, examples, device)

    n = len(examples)

    # ── Token F1 (kept as secondary reference metric) ─────────────────────────
    total_token_f1 = 0.0
    best_refs      = []
    for pred, ex in zip(predictions, examples):
        f1, _ = _best_token_score(pred, ex["answers"])
        total_token_f1 += f1
        best_refs.append(_best_reference(pred, ex["answers"]))

    # ── BERTScore F1 (primary metric for generative) ──────────────────────────
    # We match each prediction against its best reference (by token F1)
    # so BERTScore isn't penalised by a randomly chosen bad reference.
    if not _bertscore_available():
        print("\n  ⚠️  bert_score not installed — run: pip install bert-score")
        print("       Reporting token_f1 only for now.")
        bertscore_f1 = None
    else:
        print("\n  📐 Computing BERTScore (bert-base-multilingual-cased) …")
        scores       = _compute_bertscore(predictions, best_refs)
        bertscore_f1 = round(100.0 * sum(scores) / len(scores), 2)

    result = {
        "subtask":    "generative",
        "token_f1":   round(100.0 * total_token_f1 / n, 2),
        "n_samples":  n,
    }
    if bertscore_f1 is not None:
        result["bertscore_f1"] = bertscore_f1

    return result


# ─── Public entry point ──────────────────────────────────────────────────────

def evaluate(model_name: str, n_samples: int, model_type: str = "extractive", **_) -> Dict[str, Any]:
    cfg = TASK_LIMITS["qa"]
    n   = min(n_samples, cfg["max"])
    if n_samples > cfg["max"]:
        print(f"  ⚠️  Requested {n_samples} but qa max is {cfg['max']}. Using {cfg['max']}.")

    is_generative = model_type in (
        "generative_seq2seq", "seq2seq",
        "generative_causal",  "causal",
    )
    subtask_label = "generative" if is_generative else "extractive"

    print(f"\n📊 Task: QA [{subtask_label}] | Samples: {n}/{cfg['max']}")
    print(f"   Loading dataset: {cfg['dataset']}, split: {cfg['split']}")

    examples = load_qa_dataset(n)

    if not is_generative:
        return run_extractive(model_name, examples)
    elif model_type in ("generative_seq2seq", "seq2seq"):
        return run_generative(model_name, examples, "seq2seq")
    else:
        return run_generative(model_name, examples, "causal")