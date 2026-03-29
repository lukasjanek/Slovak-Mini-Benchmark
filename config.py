"""
Central configuration for Slovak Benchmark Mini.
All dataset limits, splits, seeds and scoring weights live here.
"""

RANDOM_SEED = 42

# ─── Dataset limits ───────────────────────────────────────────────────────────
# Max = hard cap (total rows available in the target split).
# Default = how many we evaluate when the user does not pass --samples.

TASK_LIMITS = {
    "qa": {
        "max": 9583,
        "default": 9583,
        "dataset": "lukasjanek/skquad",
        "split": "validation",
        "description": "SKQuAD validation set",
    },
    "classification": {
        "max": 3400,
        "default": 3400,
        "dataset": "lukasjanek/senti-sk",
        "split": "test",
        "description": "SentiSK test set",
    },
    "fill_mask": {
        "max": 3000,
        "default": 1000,
        "dataset": "lukasjanek/slovaksum",
        "config":   None,
        "split":    "validation",
        "text_col": "text",
        "description": "SlovakSum validation set (news articles)",
    },
    "generation": {
        "max": 3000,
        "default": 1000,
        "dataset": "lukasjanek/slovaksum",
        "config":  None,
        "split":   "test",
        "text_col": "text",
        "description": "SlovakSum test set (news articles)",
    },
}
# ─── Scoring weights for overall_score ────────────────────────────────────────
# Keys match task names above.  Must sum to 1.0.
# Primary metric used per task:
#   qa            → f1
#   classification → f1_weighted
#   fill_mask      → perplexity  (inverted: score = 100 / perplexity)
#   generation     → perplexity  (inverted: score = 100 / perplexity)

TASK_WEIGHTS = {
    "qa": 0.35,
    "classification": 0.35,
    "fill_mask": 0.15,
    "generation": 0.15,
}

# ─── Architecture → model type mapping ───────────────────────────────────────
# Keys are substrings matched against the 'architectures' field in config.json.
# Order matters: more specific entries first.
# config.json is fetched via AutoConfig.from_pretrained() at detection time.
ARCHITECTURE_TYPE_MAP = {
    "ForQuestionAnswering":      "extractive",
    "ForSequenceClassification": "classification",
    "ForCausalLM":               "generative_causal",
    "CausalLM":                  "generative_causal",
    "ForConditionalGeneration":  "generative_seq2seq",
    "ForSeq2SeqLM":              "generative_seq2seq",
    "ForMaskedLM":               "masked",
    "MaskedLM":                  "masked",
}

# ─── Name-based fallback hints ────────────────────────────────────────────────
# Only used when config.json can't be fetched (offline, private model, etc.)
MODEL_NAME_FALLBACK = {
    "extractive":          ["squad", "extractive", "-qa"],
    "generative_seq2seq":  ["mt5", "t5", "bart", "mbart", "seq2seq"],
    "generative_causal":   ["gpt", "mistral", "llama", "falcon", "causal", "bloom"],
    "masked":              ["bert", "roberta", "electra", "xlm", "mlm"],
    "classification":      ["sentiment", "classif", "classify"],
}