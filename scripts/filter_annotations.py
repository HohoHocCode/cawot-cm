"""Filter train.json (and optionally test_gallery.json) to entries whose image
files actually exist on disk. Useful after a partial download (--num-zips < 75).

Writes <name>_local.json next to the original.

Usage:
  python scripts/filter_annotations.py --root ./pab_data
  python scripts/filter_annotations.py --root ./pab_data --files train.json test_gallery.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import _extract_image_path
from src.utils import setup_logger

logger = setup_logger("filter_anns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./pab_data")
    ap.add_argument("--files", nargs="+",
                    default=["train.json", "test_gallery.json"],
                    help="Annotation files to filter (must contain image_path entries)")
    args = ap.parse_args()

    root = Path(args.root)
    for fname in args.files:
        in_path = root / fname
        if not in_path.exists():
            logger.warning(f"Skip {in_path} (not found)")
            continue
        with open(in_path, "r", encoding="utf-8") as f:
            anns = json.load(f)
        if isinstance(anns, dict) and "data" in anns:
            anns = anns["data"]
        kept = [a for a in anns if _extract_image_path(a, root).exists()]
        out_path = in_path.with_name(in_path.stem + "_local" + in_path.suffix)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kept, f)
        logger.info(f"{fname}: kept {len(kept)} / {len(anns)} entries → {out_path.name}")


if __name__ == "__main__":
    main()
