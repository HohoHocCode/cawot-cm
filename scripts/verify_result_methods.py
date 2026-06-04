"""Verify that a sweep result contains the expected method names.

Accepts either:
  - a records.csv path, or
  - a zip file containing outputs/eval/records.csv.

This is meant as a cheap pre-upload guard so a V2.1-only archive is not
mistaken for the published-baseline sweep.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import zipfile
from collections import Counter
from pathlib import Path


def _read_records(path: Path) -> tuple[str, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            candidates = [name for name in zf.namelist() if name.endswith("records.csv")]
            if not candidates:
                raise FileNotFoundError(f"{path} does not contain records.csv")
            name = sorted(candidates)[0]
            return name, zf.read(name).decode("utf-8-sig")
    return str(path), path.read_text(encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_path", type=Path, help="records.csv or zip containing records.csv")
    ap.add_argument(
        "--expect",
        required=True,
        help="Comma-separated expected methods, e.g. clipscore,semdedup,k_center",
    )
    args = ap.parse_args()

    source_name, text = _read_records(args.result_path)
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f"{source_name} has no records")

    methods = sorted({row.get("method", "") for row in rows if row.get("method") != "zeroshot"})
    expected = [m.strip() for m in args.expect.split(",") if m.strip()]
    missing = [m for m in expected if m not in methods]
    unexpected = [m for m in methods if m not in expected]

    combos = Counter()
    for row in rows:
        method = row.get("method", "")
        if method and method != "zeroshot":
            combos[method] += 1

    print(f"source: {source_name}")
    print(f"methods: {methods}")
    print(f"rows_by_method: {dict(sorted(combos.items()))}")

    if missing:
        print(f"ERROR: missing expected methods: {missing}", file=sys.stderr)
        if unexpected:
            print(f"present unexpected methods: {unexpected}", file=sys.stderr)
        return 1
    print("method check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
