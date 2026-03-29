"""
Generation task – evaluated on slovaksum.

"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

from config import TASK_LIMITS, RANDOM_SEED


MAX_TOKENS = 256   # max tokens per passage to keep GPU memory sane

GEN_BATCH_SIZE = 4


def _bnb_config() -> BitsAndBytesConfig | None:
    """4-bit quantization when GPU is available. Skipped on CPU."""
    if not torch.cuda.is_available():
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _collect_passages(n: int) -> List[str]:
    """Load n passages from SlovakSum test split.
    """
    cfg = TASK_LIMITS["generation"]
    ds  = load_dataset(cfg["dataset"], split=cfg["split"])
    ds  = ds.shuffle(seed=RANDOM_SEED)

    text_col = cfg.get("text_col", "text")
    passages = []
    for row in ds:
        text = (row[text_col] if text_col in row else "").strip()
        if len(text.split()) < 30:
            continue
        passages.append(text)
        if len(passages) >= n:
            break

    return passages


def _causal_perplexity(model, tokenizer, passages: List[str], device,
                       eos_reliable: bool = True) -> tuple:
    """Compute perplexity and BPC for a causal LM.

    If eos_reliable=False, forces single-passage mode regardless of whether
    the tokenizer has an EOS token defined — catches models like
    slovak-nlp/mistral-sk-7b where EOS is defined but never trained on.
    """
    eos_missing = tokenizer.eos_token_id is None

    if eos_missing or not eos_reliable:
        if not eos_reliable and not eos_missing:
            print(
                f"\n  🔵 Forcing single-passage mode — model has EOS token ID "
                f"{tokenizer.eos_token_id} but was not trained to use it."
            )
        else:
            print(
                f"\n  ⚠️  No EOS token — falling back to single-passage mode "
                f"(slower but correct)."
            )
        return _causal_perplexity_single(model, tokenizer, passages, device)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    nlls        = []
    total_chars = 0
    total_toks  = 0

    for i in tqdm(range(0, len(passages), GEN_BATCH_SIZE),
                  desc="Generation (causal PPL)", unit="batch"):
        batch = passages[i : i + GEN_BATCH_SIZE]
        enc   = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=MAX_TOKENS, padding=True,
        ).to(device)

        input_ids      = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        labels         = input_ids.clone()
        labels[attention_mask == 0] = -100

        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            out  = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss
        if not torch.isnan(loss):
            nlls.append(loss.item())
            real_toks    = attention_mask.sum().item()
            real_chars   = sum(len(t) for t in batch)
            total_toks  += real_toks
            total_chars += real_chars

    if not nlls:
        return float("inf"), float("inf")

    avg_nll       = sum(nlls) / len(nlls)
    ppl           = math.exp(avg_nll)
    chars_per_tok = total_chars / max(total_toks, 1)
    bpc           = avg_nll / math.log(2) / max(chars_per_tok, 1)
    return round(ppl, 4), round(bpc, 4)


def _causal_perplexity_single(model, tokenizer, passages: List[str], device) -> tuple:
    """Single-passage perplexity — no padding needed, works for any causal LM."""
    nlls        = []
    total_chars = 0
    total_toks  = 0

    for text in tqdm(passages, desc="Generation (causal PPL, single)", unit="doc"):
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS,
        ).to(device)
        input_ids = enc["input_ids"]
        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            out  = model(input_ids, labels=input_ids)
            loss = out.loss
        if not torch.isnan(loss):
            nlls.append(loss.item())
            total_toks  += input_ids.shape[1]
            total_chars += len(text)

    if not nlls:
        return float("inf"), float("inf")

    avg_nll       = sum(nlls) / len(nlls)
    ppl           = math.exp(avg_nll)
    chars_per_tok = total_chars / max(total_toks, 1)
    bpc           = avg_nll / math.log(2) / max(chars_per_tok, 1)
    return round(ppl, 4), round(bpc, 4)


def _masked_pseudo_perplexity(model, tokenizer, passages: List[str], device) -> tuple:
    """Pseudo-perplexity + BPC for masked LMs."""
    rng       = random.Random(RANDOM_SEED + 2)
    log_probs = []
    total_chars = 0
    total_toks  = 0

    for text in tqdm(passages, desc="Generation (MLM pseudo-PPL)", unit="doc"):
        tokens = tokenizer.tokenize(text)[:MAX_TOKENS - 2]
        if len(tokens) < 4:
            continue

        mask_positions = list(range(0, len(tokens), 5))
        for pos in mask_positions[:10]:
            orig      = tokens[pos]
            toks_copy = tokens.copy()
            toks_copy[pos] = tokenizer.mask_token
            masked_text = tokenizer.convert_tokens_to_string(toks_copy)

            inputs = tokenizer(masked_text, return_tensors="pt",
                               truncation=True, max_length=MAX_TOKENS).to(device)
            mask_token_id = tokenizer.mask_token_id
            mask_idx = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]
            if len(mask_idx) == 0:
                continue

            orig_id = tokenizer.convert_tokens_to_ids(orig)
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits[0, mask_idx[0]], dim=-1)
            p     = probs[orig_id].item()
            log_probs.append(math.log(max(p, 1e-10)))
            total_toks  += len(tokens)
            total_chars += len(text)

    if not log_probs:
        return float("inf"), float("inf")

    avg_nll       = -sum(log_probs) / len(log_probs)
    ppl           = math.exp(avg_nll)
    chars_per_tok = total_chars / max(total_toks, 1)
    bpc           = avg_nll / math.log(2) / max(chars_per_tok, 1)
    return round(ppl, 4), round(bpc, 4)


def _seq2seq_perplexity(model, tokenizer, passages: List[str], device) -> tuple:
    """Perplexity + BPC for encoder-decoder models."""
    nlls        = []
    total_chars = 0
    total_toks  = 0

    for text in tqdm(passages, desc="Generation (seq2seq PPL)", unit="doc"):
        words     = text.split()
        mid       = len(words) // 2
        src, tgt  = " ".join(words[:mid]), " ".join(words[mid:])

        enc = tokenizer(src, return_tensors="pt", truncation=True,
                        max_length=MAX_TOKENS // 2).to(device)
        dec = tokenizer(tgt, return_tensors="pt", truncation=True,
                        max_length=MAX_TOKENS // 2).to(device)

        labels            = dec["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            out  = model(**enc, decoder_input_ids=dec["input_ids"], labels=labels)
            loss = out.loss
        if not torch.isnan(loss):
            nlls.append(loss.item())
            total_toks  += dec["input_ids"].shape[1]
            total_chars += len(tgt)

    if not nlls:
        return float("inf"), float("inf")

    avg_nll       = sum(nlls) / len(nlls)
    ppl           = math.exp(avg_nll)
    chars_per_tok = total_chars / max(total_toks, 1)
    bpc           = avg_nll / math.log(2) / max(chars_per_tok, 1)
    return round(ppl, 4), round(bpc, 4)


# ─── Rating helpers ───────────────────────────────────────────────────────────
#
# BPC reference:  < 0.5 excellent · 0.5–1.0 good · 1.0–2.0 fair · > 2.0 poor
# PPL reference:  < 20  excellent · 20–50   good · 50–100  fair · > 100 poor

def _bpc_rating(bpc: float) -> str:
    if bpc == float("inf") or bpc != bpc:   # nan check
        return "n/a"
    if bpc < 0.5:   return "excellent"
    if bpc < 1.0:   return "good"
    if bpc < 2.0:   return "fair"
    return "poor"


def _ppl_rating(ppl: float) -> str:
    if ppl == float("inf") or ppl != ppl:
        return "n/a"
    if ppl < 20:    return "excellent"
    if ppl < 50:    return "good"
    if ppl < 100:   return "fair"
    return "poor"


# ─── Public entry point ──────────────────────────────────────────────────────

def evaluate(model_name: str, n_samples: int, model_type: str = "causal", **_) -> Dict[str, Any]:
    cfg = TASK_LIMITS["generation"]

    NO_EOS_CAP  = 250   # single-passage default
    NO_EOS_MAX  = 500   # hard ceiling for single-passage mode
    eos_reliable = True  # assume reliable until proven otherwise

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import json as _json

        repo_files = set(list_repo_files(model_name))

        if "generation_config.json" not in repo_files:
            pass
        else:
            gen_cfg_path = hf_hub_download(model_name, "generation_config.json")
            gen_cfg      = _json.load(open(gen_cfg_path, encoding="utf-8"))
            gen_eos      = gen_cfg.get("eos_token_id")
            auto_generated = gen_cfg.get("_from_model_config", False)

            if auto_generated and gen_eos is not None:
                # Has EOS in config but it was inherited, not trained — the key case
                eos_reliable = False
                print(
                    f"  🔵 generation_config.json was auto-generated (_from_model_config: true).\n"
                    f"     EOS token ID {gen_eos} was inherited from base model but the model\n"
                    f"     was never trained to use it — batched perplexity will be unreliable.\n"
                    f"     Using single-passage mode."
                )
            elif gen_eos is None and auto_generated:
                eos_reliable = False
                print(f"  🔵 No eos_token_id in auto-generated generation config — single-passage mode.")

    except Exception:
        pass   # can't determine — assume reliable, will fall back at runtime

    if not eos_reliable:
        effective_max     = min(NO_EOS_MAX, cfg["max"])
        effective_default = NO_EOS_CAP
        n = min(n_samples if n_samples > 0 else effective_default, effective_max)
        if n_samples > effective_max:
            print(
                f"  ⚠️  Requested {n_samples} samples but model appears to ignore EOS.\n"
                f"     Batched perplexity is unreliable for such models.\n"
                f"     Using single-passage mode, auto-capped to {effective_max} samples."
            )
        elif n_samples <= 0 or n_samples > effective_default:
            print(
                f"     Auto-set to {n} samples (reduced from {cfg['default']} "
                f"— single-passage mode is ~4x slower than batched)."
            )
    else:
        n = min(n_samples if n_samples > 0 else cfg["default"], cfg["max"])
        if n_samples > cfg["max"]:
            print(f"  ⚠️  Requested {n_samples} but generation max is {cfg['max']}. Using {cfg['max']}.")

    cfg_name = cfg.get("config") or ""
    print(f"\n📊 Task: GENERATION | Samples: {n}/{cfg['max']}")
    print(f"   Loading dataset: {cfg['dataset']}" + (f" ({cfg_name})" if cfg_name else ""))

    passages = _collect_passages(n)
    actual_n = len(passages)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading model: {model_name}  (device: {device})")

    bnb       = _bnb_config()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Confirm EOS status from actual tokenizer (pre-check may have been approximate)
    eos_missing  = tokenizer.eos_token_id is None
    eos_reliable = eos_reliable and not eos_missing

    if model_type in ("causal", "generative_causal"):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            dtype=torch.float32 if (device.type == "cpu" or bnb) else torch.float16,
            device_map="auto",
        ).eval()
        ppl, bpc = _causal_perplexity(model, tokenizer, passages, device,
                                       eos_reliable=eos_reliable)

    elif model_type in ("masked", "mlm"):
        model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            dtype=torch.float32 if (device.type == "cpu" or bnb) else torch.float16,
            device_map="auto",
        ).eval()
        ppl, bpc = _masked_pseudo_perplexity(model, tokenizer, passages, device)

    elif model_type in ("seq2seq", "generative_seq2seq"):
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            dtype=torch.float32 if (device.type == "cpu" or bnb) else torch.float16,
            device_map="auto",
        ).eval()
        ppl, bpc = _seq2seq_perplexity(model, tokenizer, passages, device)

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb,
            dtype=torch.float32 if (device.type == "cpu" or bnb) else torch.float16,
            device_map="auto",
        ).eval()
        ppl, bpc = _causal_perplexity(model, tokenizer, passages, device,
                                       eos_reliable=eos_reliable)

    return {
        "perplexity":        ppl,
        "perplexity_rating": _ppl_rating(ppl),
        "bpc":               bpc,
        "bpc_rating":        _bpc_rating(bpc),
        "reliable":          eos_reliable,
        "eval_mode":         "single_passage" if (eos_missing or not eos_reliable) else "batched",
        "n_samples":         actual_n,
    }