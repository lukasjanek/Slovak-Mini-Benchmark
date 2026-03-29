"""
leaderboard.py – four-section leaderboard, one section per task.

Each section only shows columns relevant to that task.
QA section shows a TYPE tag (extractive / generative) and the right metrics.
Empty sections still print a header so you know nothing has been run yet.

Usage:
    python leaderboard.py                     # all four sections
    python leaderboard.py --task qa           # QA section only
    python leaderboard.py --task fill_mask
    python leaderboard.py --csv               # flat CSV dump
    python leaderboard.py --results-dir /path
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Tuple


# ─── Section definitions ──────────────────────────────────────────────────────
# Each section has a title, the result key, and which (metric_key, col_header)
# pairs to display. Order here = column order in the table.

SECTIONS = {
    "qa": {
        "title":   "QA  (Question Answering)",
        "columns": [
            ("token_f1",     "TOKEN_F1"),
            ("exact_match",  "EXACT_M"),
            ("bertscore_f1", "BERT_F1"),
            ("n_samples",    "SAMPLES"),
        ],
    },
    "classification": {
        "title":   "CLASSIFICATION  (Sentiment)",
        "columns": [
            ("f1_weighted",       "F1_W"),
            ("accuracy",          "ACC"),
            ("n_classes_model",   "MDL_CLS"),
            ("n_classes_dataset", "DS_CLS"),
            ("n_samples",         "SAMPLES"),
        ],
    },
    "fill_mask": {
        "title":   "FILL MASK",
        "columns": [
            ("top1_accuracy", "TOP1"),
            ("top5_accuracy", "TOP5"),
            ("n_samples",     "SAMPLES"),
        ],
    },
    "generation": {
        "title":   "GENERATION  (Perplexity ↓  BPC ↓  lower is better)",
        "columns": [
            ("perplexity",        "PPL"),
            ("perplexity_rating", "PPL_RATING"),
            ("bpc",               "BPC"),
            ("bpc_rating",        "BPC_RATING"),
            ("n_samples",         "SAMPLES"),
        ],
    },
}

SECTION_ORDER = ["qa", "classification", "fill_mask", "generation"]


# ─── Load ─────────────────────────────────────────────────────────────────────

def load_all_results(results_dir: str = "results") -> List[Dict[str, Any]]:
    files   = sorted(glob(os.path.join(results_dir, "*.json")))
    records = []
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                records.append(json.load(fh))
        except Exception as e:
            print(f"  ⚠️  Could not load {f}: {e}")
    return records


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _fmt(v: Any, col: str) -> str:
    """Format a cell value. PPL/BPC columns get 3 decimals, others 2."""
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}" if ("PPL" in col or col == "BPC") else f"{v:.2f}"
    if isinstance(v, int):
        return str(v)
    return str(v)


def _qa_type(rec: Dict[str, Any]) -> str:
    """Return 'extractive', 'generative', or '' if QA wasn't run."""
    qa = rec.get("results", {}).get("qa")
    if not qa or "error" in qa:
        return ""
    return qa.get("subtask", "extractive")


def _cell(task_data: Dict, metric_key: str, qa_subtask: str = "") -> Any:
    """
    Get a cell value, returning 'n/a' for metrics that don't apply to
    this QA subtask (e.g. exact_match for generative, bertscore for extractive).
    """
    if qa_subtask:
        if metric_key == "exact_match" and qa_subtask == "generative":
            return "n/a"
        if metric_key == "bertscore_f1" and qa_subtask == "extractive":
            return "n/a"
    val = task_data.get(metric_key)
    return val if val is not None else "n/t"


def _section_rows(records: List[Dict], task: str) -> List[Tuple]:
    """
    Return rows for one section — only records that actually ran this task.
    Each row: (model, timestamp, qa_type, {col: value})
    """
    cols    = SECTIONS[task]["columns"]
    rows    = []

    for rec in records:
        tasks_run = set(rec.get("tasks_run", []))
        if task not in tasks_run:
            continue

        task_data = rec.get("results", {}).get(task, {})
        if not task_data or "error" in task_data:
            # Still include it so ERR is visible
            values = {col: "ERR" for _, col in cols}
        else:
            qa_sub = _qa_type(rec) if task == "qa" else ""
            values = {col: _cell(task_data, mk, qa_sub) for mk, col in cols}

        rows.append((
            rec.get("model", "?"),
            rec.get("timestamp", ""),
            _qa_type(rec) if task == "qa" else "",
            values,
        ))

    return rows


def _primary_sort_col(task: str) -> str:
    return {
        "qa":             "TOKEN_F1",
        "classification": "F1_W",
        "fill_mask":      "TOP1",
        "generation":     "PPL",      # lower is better — handled separately
    }[task]


# ─── Print one section ────────────────────────────────────────────────────────

def _print_section(task: str, rows: List[Tuple], show_empty: bool = True) -> None:
    section   = SECTIONS[task]
    col_defs  = section["columns"]
    col_names = [col for _, col in col_defs]
    title     = section["title"]

    # QA section has an extra TYPE column
    has_type  = (task == "qa")
    type_w    = 11  # len("generative") + 1

    # Column widths
    col_w   = {col: max(len(col), 7) for col in col_names}
    if rows:
        for _, _, _, vals in rows:
            for col in col_names:
                col_w[col] = max(col_w[col], len(_fmt(vals.get(col, "n/t"), col)))
    model_w = max((len(r[0]) for r in rows), default=5) + 2
    ts_w    = 17

    # Header line
    header = f"{'MODEL':<{model_w}} {'TIMESTAMP':<{ts_w}}"
    if has_type:
        header += f" {'TYPE':<{type_w}}"
    for col in col_names:
        header += f"  {col:>{col_w[col]}}"
    width = len(header)

    print(f"\n{'━' * width}")
    print(f"  📋  {title}")
    if task == "generation":
        print(f"  {'─' * (width - 2)}")
        print(f"  PPL ratings:  excellent < 20  ·  good 20–50  ·  fair 50–100  ·  poor > 100")
        print(f"  BPC ratings:  excellent < 0.5 ·  good 0.5–1  ·  fair 1–2     ·  poor > 2")
    print(f"{'━' * width}")

    if not rows:
        if show_empty:
            print(f"  {'MODEL':<{model_w}} {'TIMESTAMP':<{ts_w}}", end="")
            if has_type:
                print(f" {'TYPE':<{type_w}}", end="")
            for col in col_names:
                print(f"  {col:>{col_w[col]}}", end="")
            print()
            print(f"  {'─' * (width - 2)}")
            print(f"  (no results yet)")
        print(f"{'━' * width}")
        return

    # Sort — PPL lower is better, everything else higher is better
    sort_col = _primary_sort_col(task)
    def _key(r):
        v = r[3].get(sort_col, "n/t")
        if isinstance(v, (int, float)):
            return float(v) if task != "generation" else -float(v)
        return -1.0

    sorted_rows = sorted(rows, key=_key, reverse=True)

    print(f"  {header}")
    print(f"  {'─' * (width - 2)}")

    for model, ts, qa_type, vals in sorted_rows:
        line = f"  {model:<{model_w}} {ts:<{ts_w}}"
        if has_type:
            tag = f"[{qa_type}]" if qa_type else ""
            line += f" {tag:<{type_w}}"
        for col in col_names:
            line += f"  {_fmt(vals.get(col, 'n/t'), col):>{col_w[col]}}"
        print(line)

    print(f"{'━' * width}")


# ─── Main print ───────────────────────────────────────────────────────────────

def print_leaderboard(records: List[Dict[str, Any]],
                      task_filter: Optional[str] = None) -> None:
    tasks = [task_filter] if task_filter else SECTION_ORDER

    print("\n" + "=" * 66)
    print("🏆  SLOVAK BENCHMARK MINI – LEADERBOARD")
    print("    n/t = not tested  |  n/a = not applicable  |  ERR = crashed")
    print("=" * 66)

    for task in tasks:
        rows = _section_rows(records, task)
        _print_section(task, rows, show_empty=True)


# ─── CSV ──────────────────────────────────────────────────────────────────────

def print_csv(records: List[Dict[str, Any]],
              task_filter: Optional[str] = None) -> None:
    tasks = [task_filter] if task_filter else SECTION_ORDER

    for task in tasks:
        col_defs  = SECTIONS[task]["columns"]
        col_names = [col for _, col in col_defs]
        has_type  = (task == "qa")

        header = "task,model,timestamp"
        if has_type:
            header += ",type"
        header += "," + ",".join(col_names)
        print(header)

        for model, ts, qa_type, vals in _section_rows(records, task):
            line = f"{task},{model},{ts}"
            if has_type:
                line += f",{qa_type}"
            line += "," + ",".join(_fmt(vals.get(c, "n/t"), c) for c in col_names)
            print(line)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Slovak Benchmark Mini – Leaderboard")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--task", default=None,
                        help="Show only one section: qa | classification | fill_mask | generation")
    parser.add_argument("--csv", action="store_true", help="CSV output")
    args = parser.parse_args()

    records = load_all_results(args.results_dir)
    if args.csv:
        print_csv(records, args.task)
    else:
        print_leaderboard(records, args.task)


if __name__ == "__main__":
    main()