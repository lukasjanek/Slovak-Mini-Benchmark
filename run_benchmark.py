#!/usr/bin/env python3
"""Slovak Benchmark Mini – evaluate language models on Slovak NLP tasks."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config import TASK_LIMITS


# ─── Help ────────────────────────────────────────────────────────────────────

def print_help() -> None:
    print("""
┌─────────────────────────────────────────────────────────────────┐
│              🇸🇰  SLOVAK BENCHMARK MINI  –  CLI HELP             │
└─────────────────────────────────────────────────────────────────┘

USAGE
  python run_benchmark.py [OPTIONS]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CORE OPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --model   MODEL_ID     HuggingFace model id (required for eval)
  --task    TASK         Run a single task only
                           qa | classification | fill_mask | generation
  --mode    full         Run all 4 tasks (ignores --task)
  --samples N            How many samples per task (default = per-task max)
                           Automatically capped if N exceeds dataset limit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 MODEL TYPE OVERRIDE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --model-type TYPE      Skip auto-detection and force a model type
                           extractive        BERT/RoBERTa with QA head
                           masked            BERT/RoBERTa base (MLM)
                           generative_causal GPT / Mistral / LLaMA
                           generative_seq2seq T5 / mT5 / BART
                           classification    Sequence classification head

  Auto-detection reads config.json from HuggingFace — only use this
  flag if the model is private, offline, or detects incorrectly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 OUTPUT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --output-dir  DIR      Where to save result JSON files (default: results/)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 UTILITY COMMANDS  (no --model needed)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  --help, -h             Show this help message
  --limits               Show dataset sample limits and exit
  -i                     Interactive mode (guided prompts)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 LEADERBOARD  (separate script)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  python leaderboard.py                   Full leaderboard table
  python leaderboard.py --task qa         Filter to QA runs only
  python leaderboard.py --csv             CSV output

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 TASKS & METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  qa             SKQuAD validation    → exact_match, f1  (primary: f1)
  classification SentiSK test         → accuracy, f1_weighted, f1_macro
  fill_mask      FineWeb-2 Slovak     → token_accuracy, pseudo_perplexity
  generation     Wikipedia Slovak     → perplexity

  overall_score  Weighted average across all tested tasks
                   QA 35% + Classification 35% + Fill-mask 15% + Generation 15%
                   Only shown when more than one task was evaluated.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 EXAMPLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Show dataset limits
  python run_benchmark.py --limits

  # QA with 500 samples (auto-detects model type)
  python run_benchmark.py --model TUKE-DeutscheTelekom/slovakbert-skquad --task qa --samples 500

  # QA and force extractive type
  python run_benchmark.py --model gerulata/slovakbert --task qa --model-type extractive --samples 500

  # Full benchmark at default sample counts
  python run_benchmark.py --model gerulata/slovakbert --mode full

  # Classification only, 2000 samples, custom output dir
  python run_benchmark.py --model crabz/slovakbert-sentiment --task classification --samples 2000 --output-dir my_results

  # Interactive guided mode
  python run_benchmark.py -i
""")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def print_limits() -> None:
    print("\nDataset sample limits:")
    print(f"{'TASK':<20} {'Max':>8}   {'Default':>8}   {'Dataset'}")
    print("-" * 70)
    for task, cfg in TASK_LIMITS.items():
        print(f"{task.upper():<20} {cfg['max']:>8}   {cfg['default']:>8}   {cfg['dataset']}")
    print()


def resolve_samples(task: str, requested: int) -> int:
    limit = TASK_LIMITS[task]
    if requested <= 0:
        return limit["default"]
    if requested > limit["max"]:
        print(f"⚠️  Requested {requested} but {task} max is {limit['max']}. Using {limit['max']}.")
        return limit["max"]
    return requested


def _interactive() -> None:
    print("\n🇸🇰  Slovak Benchmark Mini – Interactive Mode")
    print_limits()

    model = input("Model name (HuggingFace id): ").strip()
    if not model:
        print("No model provided. Exiting.")
        return

    print("\nAvailable tasks:", ", ".join(TASK_LIMITS.keys()), ", full (all)")
    task_input = input("Task(s) [default: auto]: ").strip().lower()

    samples_input = input("Number of samples per task [default: per-task default]: ").strip()
    n_samples     = int(samples_input) if samples_input.isdigit() else -1

    model_type_input = input("Model type override (leave blank for auto-detect): ").strip()
    model_type       = model_type_input if model_type_input else None

    tasks = None
    if task_input and task_input != "full":
        tasks = [t.strip() for t in task_input.replace(",", " ").split() if t.strip() in TASK_LIMITS]
        if not tasks:
            print("No valid tasks specified – running auto-detected tasks.")
            tasks = None

    from evaluation import run_evaluation
    run_evaluation(
        model_name=model,
        tasks=tasks,
        n_samples=n_samples,
        model_type=model_type,
    )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Slovak Benchmark Mini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,  # we handle --help ourselves for custom formatting
    )

    parser.add_argument("--model",       type=str, default=None)
    parser.add_argument("--task",        type=str, default=None)
    parser.add_argument("--mode",        type=str, default=None)
    parser.add_argument("--samples",     type=int, default=-1)
    parser.add_argument("--model-type",  type=str, default=None, dest="model_type")
    parser.add_argument("--output-dir",  type=str, default="results", dest="output_dir")
    parser.add_argument("--limits",      action="store_true")
    parser.add_argument("-i",            action="store_true", dest="interactive")
    parser.add_argument("--help", "-h",  action="store_true", dest="show_help")

    args = parser.parse_args()

    if args.show_help or len(sys.argv) == 1:
        print_help()
        return

    if args.limits:
        print_limits()
        return

    if args.interactive:
        _interactive()
        return

    if not args.model:
        print_help()
        print("❌  --model is required (unless using --limits or -i)\n")
        sys.exit(1)

    tasks = None
    if args.task:
        if args.task not in TASK_LIMITS:
            print(f"❌  Unknown task '{args.task}'. Choose from: {list(TASK_LIMITS.keys())}")
            sys.exit(1)
        n = resolve_samples(args.task, args.samples)
        tasks = [args.task]
        args.samples = n
    elif args.mode == "full":
        tasks = list(TASK_LIMITS.keys())

    from evaluation import run_evaluation
    run_evaluation(
        model_name=args.model,
        tasks=tasks,
        n_samples=args.samples,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

