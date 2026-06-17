#!/usr/bin/env python
"""Generate V3 Q_proxy families with DeepSeek V4 Pro.

Examples
--------
Smoke comparison:
    PYTHONPATH=. python scripts/generate_qproxy_v3.py \\
        --mode both --n-calls 2 --n-per-family 8 --out outputs/qproxy_v3_smoke

Full non-thinking run:
    PYTHONPATH=. python scripts/generate_qproxy_v3.py \\
        --mode nonthinking --n-calls 40 --n-per-family 30 --resume \\
        --target-total-queries 20000 --out outputs/qproxy_v3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cawot.qproxy_v3 import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_DEFAULT_MODEL,
    ProgressEvent,
    compare_mode_outputs,
    family_names_for_preset,
    generate_qproxy_families,
    load_api_key,
    save_family_outputs,
)


class ConsoleProgress:
    def __init__(self, *, enabled: bool = True, stream=None):
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self._last_len = 0
        self._finished = False

    def __call__(self, event: ProgressEvent) -> None:
        if not self.enabled:
            return
        target = max(event.target_queries, 1)
        completed = min(event.completed_queries, event.target_queries)
        ratio = completed / target
        width = 26
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        resumed = f" resumed={event.resumed_calls}" if event.resumed_calls else ""
        line = (
            f"{event.mode} qproxy [{bar}] "
            f"{event.completed_queries}/{event.target_queries} queries "
            f"({ratio * 100:5.1f}%) "
            f"batches {event.completed_batches}/{event.total_batches}"
            f"{resumed}"
        )
        padding = " " * max(0, self._last_len - len(line))
        self.stream.write("\r" + line + padding)
        self.stream.flush()
        self._last_len = len(line)
        if event.completed_batches >= event.total_batches:
            self.stream.write("\n")
            self.stream.flush()
            self._finished = True

    def finish(self) -> None:
        if self.enabled and not self._finished:
            self.stream.write("\n")
            self.stream.flush()
            self._finished = True


def _run_one_mode(args, api_key: str, mode: str, out_dir: Path):
    progress = ConsoleProgress(enabled=not args.no_progress)
    try:
        families, stats = generate_qproxy_families(
            api_key=api_key,
            mode=mode,
            n_calls=args.n_calls,
            n_per_family=args.n_per_family,
            max_workers=args.max_workers,
            model=args.model,
            base_url=args.base_url,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            temperature_jitter=args.temperature_jitter,
            seed=args.seed,
            max_retries=args.max_retries,
            family_preset=args.family_preset,
            annotation_dir=args.annotations_dir,
            max_q1_queries=args.max_q1_queries,
            checkpoint_dir=out_dir / ".checkpoints" if args.resume else None,
            progress_callback=progress if not args.no_progress else None,
            target_total_queries=args.target_total_queries,
            max_topup_calls=args.max_topup_calls,
            min_query_words=args.min_query_words,
            cross_family_dedupe=not args.no_cross_family_dedupe,
        )
    finally:
        progress.finish()
    save_family_outputs(families, stats, out_dir)
    logging.info(
        "%s done: raw=%d unique=%d dup_rate=%.1f%% failed_calls=%d",
        mode,
        stats.raw_query_count,
        stats.unique_query_count,
        stats.duplicate_rate * 100,
        stats.failed_calls,
    )
    return families, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate CAWOT V3 Q_proxy families.")
    ap.add_argument("--mode", choices=("thinking", "nonthinking", "both"), default="both",
                    help="Generation mode. 'both' writes separate outputs and a comparison report.")
    ap.add_argument("--family-preset", choices=("default", "hybrid"), default="default",
                    help="Proxy family set. 'hybrid' builds Q1 from annotations and asks DeepSeek for Q2-Q6.")
    ap.add_argument("--annotations-dir", default=None,
                    help="Directory containing PAB train annotation JSON/JSONL files. Required for --family-preset hybrid.")
    ap.add_argument("--max-q1-queries", type=int, default=400,
                    help="Maximum local Q1 template queries to keep for the hybrid preset.")
    ap.add_argument("--api-key", default=None,
                    help="DeepSeek API key passed directly. Prefer --api-key-file or env for shell history.")
    ap.add_argument("--api-key-file", default=None,
                    help="File containing the DeepSeek API key. Defaults also check DEEPSEEK_API_KEY and deepseek_api.txt.")
    ap.add_argument("--model", default=DEEPSEEK_DEFAULT_MODEL,
                    help="DeepSeek model name. Default: deepseek-v4-pro.")
    ap.add_argument("--base-url", default=DEEPSEEK_BASE_URL,
                    help="OpenAI-compatible DeepSeek base URL.")
    ap.add_argument("--n-calls", type=int, default=10,
                    help="API calls per mode.")
    ap.add_argument("--n-per-family", type=int, default=25,
                    help="Queries requested for each family per call.")
    ap.add_argument("--max-workers", type=int, default=2,
                    help="Parallel API calls per mode.")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="Maximum output tokens per call.")
    ap.add_argument("--max-retries", type=int, default=4,
                    help="Retries per failed/empty batch.")
    ap.add_argument("--temperature", type=float, default=0.85,
                    help="Sampling temperature for non-thinking mode only.")
    ap.add_argument("--temperature-jitter", type=float, default=0.1,
                    help="Per-batch temperature jitter for non-thinking mode only.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for deterministic batch temperatures.")
    ap.add_argument("--out", default="outputs/qproxy_v3",
                    help="Output directory.")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse successful batch checkpoints from the output directory.")
    ap.add_argument("--target-total-queries", type=int, default=None,
                    help="Post-process and top up until this many final unique queries are available.")
    ap.add_argument("--max-topup-calls", type=int, default=32,
                    help="Maximum extra API calls for --target-total-queries.")
    ap.add_argument("--min-query-words", type=int, default=8,
                    help="Drop final qproxy queries shorter than this many words.")
    ap.add_argument("--no-cross-family-dedupe", action="store_true",
                    help="Keep exact duplicate query strings across different proxy families.")
    ap.add_argument("--no-progress", action="store_true",
                    help="Disable the qproxy query progress bar.")
    ap.add_argument("--verbose", action="store_true",
                    help="Enable debug logs.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = load_api_key(args.api_key_file, direct_key=args.api_key)
    out = Path(args.out)

    if args.mode == "both":
        thinking_families, thinking_stats = _run_one_mode(args, api_key, "thinking", out / "thinking")
        nonthinking_families, nonthinking_stats = _run_one_mode(args, api_key, "nonthinking", out / "nonthinking")
        comparison = {
            "model": args.model,
            "thinking": thinking_stats.to_dict(),
            "nonthinking": nonthinking_stats.to_dict(),
            "overlap": compare_mode_outputs(
                thinking_families,
                nonthinking_families,
                family_names_for_preset(args.family_preset),
            ),
            "notes": [
                "DeepSeek thinking mode ignores temperature/top_p; nonthinking mode uses temperature jitter.",
                "Use parse success, duplicate rate, per-family balance, and overlap before choosing a main proxy set.",
            ],
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison_report.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logging.info("Wrote comparison report to %s", out / "comparison_report.json")
    else:
        _run_one_mode(args, api_key, args.mode, out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
