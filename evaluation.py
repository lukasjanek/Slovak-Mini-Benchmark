from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import TASK_LIMITS, TASK_WEIGHTS, ARCHITECTURE_TYPE_MAP, MODEL_NAME_FALLBACK


# ─── Model-type detection ────────────────────────────────────────────────────

def _detect_from_config_json(model_name: str) -> Optional[str]:
    """
    Fetch the model's config.json from HuggingFace and read the
    'architectures' field.
    Patterns are sorted longest-first so more specific suffixes win over
    shorter ones that could be substrings of each other.
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        architectures = getattr(cfg, "architectures", None) or []

        # Sort patterns by length descending — most specific suffix wins
        sorted_patterns = sorted(ARCHITECTURE_TYPE_MAP.items(),
                                 key=lambda x: len(x[0]), reverse=True)

        for arch in architectures:
            for pattern, model_type in sorted_patterns:
                if arch.endswith(pattern):
                    print(f"  🔍 Detected architecture: {arch} → type: {model_type}")
                    return model_type

        if architectures:
            print(f"  ⚠️  Unrecognised architecture(s): {architectures} — falling back to name hints.")
        return None

    except Exception as e:
        print(f"  ⚠️  Could not fetch config.json ({e}) — falling back to name hints.")
        return None


def _detect_from_name(model_name: str) -> str:
    """Last-resort fallback: scan the model name for known keyword hints."""
    name_lower = model_name.lower()
    for mtype, keywords in MODEL_NAME_FALLBACK.items():
        if any(kw in name_lower for kw in keywords):
            return mtype
    # Final default: most Slovak community models are BERT-style masked LMs
    return "masked"


def _detect_model_type(model_name: str, explicit_type: Optional[str] = None) -> str:
    if explicit_type:
        print(f"  ✅ Model type set manually: {explicit_type}")
        return explicit_type.lower()

    detected = _detect_from_config_json(model_name)
    if detected:
        return detected

    detected = _detect_from_name(model_name)
    print(f"  🔍 Detected type from model name: {detected}")
    return detected


def _tasks_for_type(model_type: str) -> List[str]:
    """Return which tasks make sense for this model type.

    fill_mask is ONLY valid for masked LMs — BERT/RoBERTa style models
    that were pretrained with a [MASK] token. Causal and seq2seq models
    don't have one and will crash if routed there.
    """
    TASK_MAP = {
        "extractive":         ["qa"],
        "generative_causal":  ["generation"],
        "generative_seq2seq": ["qa", "generation"],
        "masked":             ["fill_mask", "classification"],
        "classification":     ["classification"],
    }
    return TASK_MAP.get(model_type, list(TASK_LIMITS.keys()))


# ─── Overall score ───────────────────────────────────────────────────────────

OVERALL_TASKS = {"qa", "classification"}

def _primary_metric(task: str, result: Dict) -> Optional[float]:
    """
    Extract the single number used for overall_score.
    Returns None for fill_mask and generation — they don't contribute.
    """
    if task == "qa":
        if result.get("subtask") == "generative":
            return result.get("bertscore_f1") or result.get("token_f1")
        return result.get("token_f1")

    if task == "classification":
        return result.get("f1_weighted")

    return None


def compute_overall_score(all_results: Dict[str, Any]) -> Optional[float]:
    """
    Weighted average of QA and Classification F1 only.
    Returns None if neither task was evaluated (e.g. generation-only run).
    """
    scores  = []
    weights = []

    for task in OVERALL_TASKS:
        result = all_results.get(task)
        if not result or "error" in result:
            continue
        metric = _primary_metric(task, result)
        if metric is None:
            continue
        scores.append(metric)
        weights.append(TASK_WEIGHTS.get(task, 1.0))

    if not scores:
        return None   # no eligible tasks ran

    total = sum(s * w for s, w in zip(scores, weights))
    return round(total / sum(weights), 2)


def print_results(model_name: str, results: Dict[str, Any]) -> None:
    print("\n" + "=" * 50)
    print("📊 RESULTS SUMMARY")
    print("=" * 50)

    for task, data in results.items():
        if task == "overall_score":
            continue
        print(f"\n📊 {task.upper()}:")
        if "error" in data:
            print(f"   ❌ Error: {data['error']}")
        else:
            for k, v in data.items():
                print(f"   {k}: {v}")

    if "overall_score" in results:
        score = results["overall_score"]
        if score is None:
            print(f"\n🏆 OVERALL SCORE: n/a  (only QA + Classification contribute)")
        else:
            print(f"\n🏆 OVERALL SCORE: {score}  (QA + Classification F1 average)")
    print("=" * 50)


# ─── Save ────────────────────────────────────────────────────────────────────

def save_results(model_name: str, results: Dict[str, Any], output_dir: str = "results") -> str:
    os.makedirs(output_dir, exist_ok=True)
    safe_name  = model_name.replace("/", "__")
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path   = os.path.join(output_dir, f"{safe_name}_{timestamp}.json")

    tasks_run = [t for t in results if t != "overall_score" and "error" not in results[t]]

    payload = {
        "model":      model_name,
        "timestamp":  timestamp,
        "tasks_run":  tasks_run,   # which tasks were actually evaluated
        "results":    results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved to: {out_path}")
    return out_path


def _check_tokenizer_compatibility(model_name: str) -> Optional[str]:
    """
    Scan the model repo before downloading any weights.
    Steps in order — stops at the first problem found:
      1. List all files in the repo
      2. If no tokenizer files at all → incompatible, stop
      3. Download tokenizer_config.json and check for known custom classes
      4. Try AutoTokenizer.from_pretrained() + encode a test string
    """
    try:
        from huggingface_hub import list_repo_files, hf_hub_download
        import json

        # ── Step 1: list repo files ───────────────────────────────────────────
        try:
            repo_files = set(list_repo_files(model_name))
        except Exception as e:
            return f"Could not access repo '{model_name}': {e}"

        TOKENIZER_FILES = {
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "vocab.json",
            "spm.model",
            "sentencepiece.bpe.model",
        }
        found_tok_files = TOKENIZER_FILES & repo_files

        if not found_tok_files:
            return (
                f"No tokenizer files found in repo '{model_name}'.\n"
                f"  Repo contains: {sorted(repo_files)}\n"
                f"  A model without a tokenizer in its repo requires a separate "
                f"tokenizer to be loaded manually and cannot be benchmarked automatically.\n"
                f"  Tip: check the model card — it may reference a tokenizer from "
                f"a different repo (e.g. daviddrzik/SK_Morph_BLM family)."
            )

        # ── Step 2: read tokenizer_config.json if present ────────────────────
        if "tokenizer_config.json" in repo_files:
            try:
                path    = hf_hub_download(model_name, "tokenizer_config.json")
                tok_cfg = json.load(open(path, encoding="utf-8"))
                tok_class = tok_cfg.get("tokenizer_class", "") or ""

                INCOMPATIBLE_CLASSES = ["SKMorfoTokenizer", "SKMT", "SKMorfo"]
                for cls in INCOMPATIBLE_CLASSES:
                    if cls in tok_class:
                        return (
                            f"Model uses custom tokenizer class '{tok_class}' "
                            f"which is not compatible with AutoTokenizer.\n"
                            f"  This tokenizer comes from an external library "
                            f"and has a different API — it cannot be used here."
                        )
            except Exception:
                pass  # couldn't read it, let AutoTokenizer try below

        # ── Step 3: try loading and encoding ─────────────────────────────────
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        tokenizer("test veta")
        return None

    except Exception as e:
        return (
            f"Could not load tokenizer for '{model_name}': {e}"
        )


# ─── Main orchestrator ───────────────────────────────────────────────────────

def run_evaluation(
    model_name:   str,
    tasks:        Optional[List[str]] = None,
    n_samples:    int                 = -1,
    model_type:   Optional[str]       = None,
    output_dir:   str                 = "results",
) -> Dict[str, Any]:

    # Dynamic imports (avoid loading torch/transformers at import time)
    sys.path.insert(0, os.path.dirname(__file__))
    from tasks import qa, classification, fill_mask, generation

    # ── Tokenizer compatibility check ─────────────────────────────────────────
    tokenizer_err = _check_tokenizer_compatibility(model_name)
    if tokenizer_err:
        print(f"\n❌  INCOMPATIBLE MODEL\n  {tokenizer_err}")
        print(
            f"\n  💡 If this model uses a custom tokenizer, you need to write\n"
            f"     a model-specific wrapper. See README.md for guidance."
        )
        return {"error": tokenizer_err}

    TASK_RUNNERS = {
        "qa":             qa.evaluate,
        "classification": classification.evaluate,
        "fill_mask":      fill_mask.evaluate,
        "generation":     generation.evaluate,
    }

    mtype = _detect_model_type(model_name, model_type)
    print(f"\n🚀 Evaluating: {model_name}")
    print(f"   Type: {mtype} | Tasks: {tasks or _tasks_for_type(mtype)}")

    active_tasks = tasks if tasks else _tasks_for_type(mtype)
    all_results: Dict[str, Any] = {}

    for task in active_tasks:
        if task not in TASK_RUNNERS:
            print(f"  ⚠️  Unknown task '{task}' – skipping.")
            continue

        limit = TASK_LIMITS[task]
        n     = n_samples if n_samples > 0 else limit["default"]
        n     = min(n, limit["max"])

        try:
            result = TASK_RUNNERS[task](
                model_name=model_name,
                n_samples=n,
                model_type=mtype,
            )
            all_results[task] = result
        except Exception as e:
            print(f"  ❌ Task '{task}' failed: {e}")
            all_results[task] = {"error": str(e)}

    all_results["overall_score"] = compute_overall_score(all_results)
    print_results(model_name, all_results)
    save_results(model_name, all_results, output_dir)

    return all_results