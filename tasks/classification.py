"""
Classification task – sentiment analysis on SentiSK.

Uses AutoModelForSequenceClassification directly (no pipeline).

Class count detection
---------------------
Before loading weights we read config.json to check how many classes
the model has. This lets us:
  - 3-class model + 3-class dataset → run as-is
  - 2-class model + 3-class dataset → automatically filter out 'neutral'
    samples and score only on positive/negative. Fair and honest.
  - Other mismatches → partial alignment with warning

Alignment strategy (in priority order):
  1. Meaningful name overlap (model labels match dataset label names)
  2. Same class count → align by sorted order
  3. 2-class model on 3-class dataset → filter neutrals from dataset
  4. Partial alignment with exclusion of unmappable samples

Metrics returned:
    accuracy           – raw accuracy
    f1_weighted        – weighted F1 (primary scorer)
    n_classes_model    – how many classes the model has
    n_classes_dataset  – how many classes the dataset has (after filtering)
    n_samples          – samples evaluated
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

from config import TASK_LIMITS, RANDOM_SEED

# Labels considered neutral — filtered out for 2-class models
NEUTRAL_LABELS = {"neutral", "neutrálny", "neutrálne", "neu", "2"}


def _normalise(s: Any) -> str:
    return str(s).lower().strip()


# ─── Class count detection ───────────────────────────────────────────────────

def _get_model_class_count(model_name: str) -> Tuple[int, Dict[int, str]]:
    """Read class count from config.json — no weights downloaded."""
    config   = AutoConfig.from_pretrained(model_name)
    id2label = getattr(config, "id2label", {0: "LABEL_0", 1: "LABEL_1"})
    return len(id2label), id2label


# ─── Dataset loader ──────────────────────────────────────────────────────────

def _load_dataset(n: int, filter_neutral: bool = False):
    """
    Load SentiSK. If filter_neutral=True, remove neutral samples so a
    2-class model is evaluated fairly on only positive/negative examples.
    """
    cfg = TASK_LIMITS["classification"]
    try:
        ds = load_dataset(cfg["dataset"], split=cfg["split"])
    except Exception:
        print(f"  ⚠️  Split '{cfg['split']}' not found — trying 'validation'.")
        ds = load_dataset(cfg["dataset"], split="validation")

    text_col  = next((c for c in ["text", "sentence", "review", "content"]
                      if c in ds.column_names), ds.column_names[0])
    label_col = next((c for c in ["label", "sentiment", "class"]
                      if c in ds.column_names), ds.column_names[-1])

    print(f"   Columns — text: '{text_col}', label: '{label_col}'")

    if filter_neutral:
        before = len(ds)
        ds = ds.filter(lambda row: _normalise(row[label_col]) not in NEUTRAL_LABELS)
        print(f"   🔵 Filtered out neutral samples: {before} → {len(ds)} remaining")

    ds = ds.shuffle(seed=RANDOM_SEED).select(range(min(n, len(ds))))

    return ds[text_col], ds[label_col], len(ds), text_col, label_col


# ─── Label merging rules ─────────────────────────────────────────────────────
# Maps fine-grained model label names → coarse dataset label names.
# Applied when model has more classes than dataset and label names suggest
# a finer-grained version of the same sentiment scale.
# Keys are normalised model label names, values are target dataset label names.

MERGE_RULES = {
    # 5-class → 3-class sentiment merging
    "very negative": "negative",
    "very positive": "positive",
    "strongly negative": "negative",
    "strongly positive": "positive",
    # Slovak variants
    "veľmi negatívny": "negative",
    "veľmi pozitívny": "positive",
    "negatívny": "negative",
    "pozitívny": "positive",
    "neutrálny": "neutral",
    # 5-star → 3-class merging (1-2 star = negative, 3 = neutral, 4-5 = positive)
    "1 star": "negative",
    "2 stars": "negative",
    "3 stars": "neutral",
    "4 stars": "positive",
    "5 stars": "positive",
}


def _try_merge(id2label: Dict[int, str],
               true_labels: List[Any]) -> Tuple[Optional[Dict], str]:
    """
    Try to merge fine-grained model labels into coarse dataset labels.
    Returns (mapping, description) or (None, "") if merge isn't applicable.
    """
    unique_true      = sorted(set(true_labels), key=_normalise)
    true_labels_norm = {_normalise(t): t for t in unique_true}

    mapping   = {}
    merged    = []
    unresolved = []

    for idx, raw in id2label.items():
        norm = _normalise(raw)
        if norm in true_labels_norm:
            # Direct match
            mapping[idx] = true_labels_norm[norm]
        elif norm in MERGE_RULES:
            target_norm = _normalise(MERGE_RULES[norm])
            if target_norm in true_labels_norm:
                mapping[idx] = true_labels_norm[target_norm]
                merged.append(f"'{raw}' → '{MERGE_RULES[norm]}'")
            else:
                unresolved.append(raw)
        else:
            unresolved.append(raw)

    if unresolved:
        return None, ""   # merge didn't cover all classes, skip

    desc = f"label merging ({len(id2label)} → {len(unique_true)} classes)"
    if merged:
        desc += f": {', '.join(merged)}"
    return mapping, desc


# ─── Label alignment ─────────────────────────────────────────────────────────

def _build_alignment(id2label: Dict[int, str],
                     true_labels: List[Any]) -> Tuple[Dict, str]:
    """
    Returns (pred_id → true_label mapping, strategy description).

    Priority order:
      1. Meaningful name overlap (e.g. model says "negative", dataset says "negative")
      2. Same class count → align by sorted order
      3. Label merging (e.g. "Very Negative" → "negative", "5 stars" → "positive")
      4. Partial — map what we can, exclude the rest
    """
    unique_true        = sorted(set(true_labels), key=_normalise)
    n_model            = len(id2label)
    n_dataset          = len(unique_true)
    model_labels_norm  = {_normalise(v): v for v in id2label.values()}
    true_labels_norm   = {_normalise(t): t for t in unique_true}

    # Strategy 1: meaningful name overlap
    has_generic = all("label_" in k for k in model_labels_norm)
    overlap     = set(model_labels_norm) & set(true_labels_norm)
    if overlap and not has_generic:
        mapping = {}
        for idx, raw in id2label.items():
            norm = _normalise(raw)
            if norm in true_labels_norm:
                mapping[idx] = true_labels_norm[norm]
        if mapping:
            return mapping, f"name overlap ({len(mapping)}/{n_model} matched)"

    # Strategy 2: same class count → sorted order
    if n_model == n_dataset:
        sorted_ids  = sorted(id2label.keys())
        sorted_true = sorted(unique_true, key=_normalise)
        mapping     = {idx: true for idx, true in zip(sorted_ids, sorted_true)}
        return mapping, f"sorted order ({n_model} classes each)"

    # Strategy 3: label merging (fine → coarse)
    if n_model > n_dataset:
        merge_mapping, merge_desc = _try_merge(id2label, true_labels)
        if merge_mapping:
            return merge_mapping, merge_desc

    # Strategy 4: partial — map what we can
    sorted_ids  = sorted(id2label.keys())
    sorted_true = sorted(unique_true, key=_normalise)
    mapping     = {idx: true for idx, true in zip(sorted_ids, sorted_true[:n_model])}
    direction   = "fewer" if n_model < n_dataset else "more"
    return mapping, (
        f"partial — model has {direction} classes ({n_model}) "
        f"than dataset ({n_dataset})"
    )


# ─── Evaluation ──────────────────────────────────────────────────────────────

def evaluate(model_name: str, n_samples: int, **_) -> Dict[str, Any]:
    cfg = TASK_LIMITS["classification"]
    n   = min(n_samples, cfg["max"])
    if n_samples > cfg["max"]:
        print(f"  ⚠️  Requested {n_samples} but classification max is {cfg['max']}. Using {cfg['max']}.")

    print(f"\n📊 Task: CLASSIFICATION | Samples: {n}/{cfg['max']}")
    print(f"   Loading dataset: {cfg['dataset']}, split: {cfg['split']}")

    # ── Check model class count BEFORE loading weights ────────────────────────
    n_model_classes, id2label = _get_model_class_count(model_name)
    print(f"   Model has {n_model_classes} classes: {dict(id2label)}")

    # Decide whether to filter neutrals
    filter_neutral = (n_model_classes == 2)
    if filter_neutral:
        print(f"   🔵 2-class model detected — will filter neutral samples from dataset")

    texts, true_labels, actual_n, text_col, label_col = _load_dataset(
        n, filter_neutral=filter_neutral
    )

    n_dataset_classes = len(set(true_labels))
    print(f"   Dataset has {n_dataset_classes} classes: {sorted(set(true_labels), key=_normalise)}")

    # Build alignment
    alignment, strategy = _build_alignment(id2label, true_labels)
    print(f"   Alignment strategy: {strategy}")
    print(f"   Alignment map: {alignment}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"  Loading model: {model_name}")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model     = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).eval()

    # ── Inference ─────────────────────────────────────────────────────────────
    BATCH        = 32
    all_pred_ids = []

    for i in tqdm(range(0, len(texts), BATCH), desc="Classification", unit="batch"):
        batch  = texts[i : i + BATCH]
        inputs = tokenizer(
            batch, return_tensors="pt", truncation=True,
            max_length=512, padding=True,
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        all_pred_ids.extend(logits.argmax(dim=-1).tolist())

    # ── Map to true label space ───────────────────────────────────────────────
    paired_true = []
    paired_pred = []
    unmapped    = 0

    for pred_id, true in zip(all_pred_ids, true_labels):
        if pred_id not in alignment:
            unmapped += 1
            continue
        paired_pred.append(alignment[pred_id])
        paired_true.append(true)

    if unmapped:
        print(f"\n  ⚠️  {unmapped} predictions had no alignment mapping and were excluded.")

    print(f"\n   Sample true:      {paired_true[:5]}")
    print(f"   Sample predicted: {paired_pred[:5]}")

    if not paired_true:
        return {
            "accuracy":           0.0,
            "f1_weighted":        0.0,
            "n_classes_model":    n_model_classes,
            "n_classes_dataset":  n_dataset_classes,
            "n_samples":          0,
        }

    accuracy    = accuracy_score(paired_true, paired_pred)
    f1_weighted = f1_score(paired_true, paired_pred, average="weighted", zero_division=0)

    return {
        "accuracy":           round(100.0 * accuracy,    2),
        "f1_weighted":        round(100.0 * f1_weighted, 2),
        "n_classes_model":    n_model_classes,
        "n_classes_dataset":  n_dataset_classes,
        "n_samples":          len(paired_true),
    }