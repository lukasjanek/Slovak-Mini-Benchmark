"""
Fill-mask task – evaluated on lukasjanek/slovaksum (validation split).

Strategy
--------
For each document in the dataset we split it into individual sentences,
then for each sentence:
  1. Tokenise with the model's tokenizer (truncated to max_length).
  2. Randomly mask one non-special, non-subword token.
  3. Run the model forward pass and get the probability of the original token.
  4. Record whether top-1 and top-5 predictions match (token accuracy).

"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM

from config import TASK_LIMITS, RANDOM_SEED


# ─── Sentence splitter ───────────────────────────────────────────────────────

def _split_sentences(text: str, min_words: int = 8) -> List[str]:
    """Split a document into sentences, keeping only those long enough."""
    # Simple split on sentence-ending punctuation followed by whitespace
    raw       = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw if len(s.strip().split()) >= min_words]

    # Fallback: use whole text if no sentences found
    if not sentences and len(text.split()) >= min_words:
        sentences = [text.strip()]

    return sentences


# ─── Dataset loader ──────────────────────────────────────────────────────────

def _collect_sentences(n: int) -> List[str]:
    """Load slovaksum validation split and extract n sentences from articles."""
    cfg      = TASK_LIMITS["fill_mask"]
    text_col = cfg.get("text_col", "text")

    try:
        ds = load_dataset(cfg["dataset"], split=cfg["split"])
    except Exception:
        print(f"  ⚠️  Split '{cfg['split']}' not found — trying 'train'.")
        ds = load_dataset(cfg["dataset"], split="train")

    ds = ds.shuffle(seed=RANDOM_SEED)

    sentences = []
    for row in ds:
        # Use direct key access — HuggingFace rows are dicts
        text = (row[text_col] if text_col in row else "").strip()
        if not text:
            continue
        sents = _split_sentences(text)
        sentences.extend(sents)
        if len(sentences) >= n * 2:
            break

    if not sentences:
        raise RuntimeError(
            f"No sentences collected from {cfg['dataset']} "
            f"(split={cfg['split']}, text_col={text_col}). "
            f"Check that the column name is correct."
        )

    random.Random(RANDOM_SEED + 1).shuffle(sentences)
    return sentences[:n]


# ─── Token normalization ─────────────────────────────────────────────────────

def _normalize(token: str) -> str:
    return token.lower().lstrip("▁").strip()


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model_name: str, n_samples: int, model_type: str = "masked", **_) -> Dict[str, Any]:
    # Fill-mask only works for masked LMs
    INCOMPATIBLE = ("generative_seq2seq", "seq2seq", "generative_causal", "causal")
    if model_type in INCOMPATIBLE:
        raise ValueError(
            f"fill_mask is not compatible with model type '{model_type}'.\n"
            f"  T5/mT5/BART (seq2seq) and GPT/Mistral (causal) models have no [MASK] token.\n"
            f"  fill_mask requires a masked LM like BERT, RoBERTa, or SlovakBERT.\n"
            f"  → Try --task generation instead, which works for any model type."
        )

    cfg = TASK_LIMITS["fill_mask"]
    n   = min(n_samples, cfg["max"])
    if n_samples > cfg["max"]:
        print(f"  ⚠️  Requested {n_samples} but fill_mask max is {cfg['max']}. Using {cfg['max']}.")

    print(f"\n📊 Task: FILL_MASK | Samples: {n}/{cfg['max']}")
    print(f"   Loading dataset: {cfg['dataset']}, split: {cfg['split']}")

    sentences = _collect_sentences(n)

    print(f"  Loading fill-mask model: {model_name}")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModelForMaskedLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).eval()

    # Get the model's actual max sequence length
    max_length = getattr(tokenizer, "model_max_length", 512)
    if max_length > 10_000:   # some tokenizers report absurdly large values
        max_length = 512

    rng        = random.Random(RANDOM_SEED + 2)
    top1       = 0
    top5       = 0
    evaluated  = 0
    errors     = 0

    # Sanity check mask token exists
    if tokenizer.mask_token_id is None:
        raise ValueError(
            f"Tokenizer for '{model_name}' has no mask_token_id — "
            f"this model cannot be used for fill-mask."
        )

    for sent in tqdm(sentences, desc="Fill-Mask", unit="sent"):

        # Tokenize with truncation BEFORE masking
        encoding  = tokenizer(
            sent,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"][0]
        attn_mask = encoding["attention_mask"][0]

        # Any non-special token is a masking candidate
        special_ids   = set(tokenizer.all_special_ids)
        candidate_idx = [
            i for i, tid in enumerate(input_ids.tolist())
            if tid not in special_ids
        ]

        if len(candidate_idx) < 3:
            continue

        mask_pos = rng.choice(candidate_idx)
        orig_id  = input_ids[mask_pos].item()

        masked_ids           = input_ids.clone()
        masked_ids[mask_pos] = tokenizer.mask_token_id

        try:
            with torch.no_grad():
                logits = model(
                    input_ids=masked_ids.unsqueeze(0).to(device),
                    attention_mask=attn_mask.unsqueeze(0).to(device),
                ).logits                                # (1, seq_len, vocab)

            probs    = torch.softmax(logits[0, mask_pos], dim=-1)
            top5_ids = probs.topk(5).indices.tolist()

            if top5_ids[0] == orig_id:
                top1 += 1
            if orig_id in top5_ids:
                top5 += 1

            evaluated += 1

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"\n  ⚠️  Forward pass error (sample {errors}): {e}")
            continue

    if errors > 0:
        print(f"\n  ⚠️  {errors}/{len(sentences)} samples failed during forward pass")

    if evaluated == 0:
        return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "n_samples": 0}

    return {
        "top1_accuracy": round(100.0 * top1 / evaluated, 2),
        "top5_accuracy": round(100.0 * top5 / evaluated, 2),
        "n_samples":     evaluated,
    }