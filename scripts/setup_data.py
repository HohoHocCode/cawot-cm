"""Download + extract N PAB shards from HuggingFace.

Source: https://huggingface.co/datasets/TruongVox/Cawot-dataset

Produces a clean local layout:

    <output>/
    ├── images/
    │   ├── imgs_0/
    │   │   ├── goal/
    │   │   │   ├── 0.jpg
    │   │   │   └── ...
    │   │   └── ...
    │   ├── imgs_1/
    │   └── ...
    └── annotations/
        ├── imgs_0.json
        ├── imgs_1.json
        └── ...

Each shard:
  - imgs_N.zip   ≈ 1.4 GB (HuggingFace)
  - imgs_N.json  ≈ 5 MB

Usage:
    # First N shards (0..N-1)
    python scripts/setup_data.py --output ./pab_data --num-shards 5
    # Specific shards
    python scripts/setup_data.py --output ./pab_data --shards 0,1,2,3,4
    # Keep zips after extraction (debugging)
    python scripts/setup_data.py --output ./pab_data --num-shards 5 --no-cleanup

The script is idempotent: skips shards whose extracted folder already exists.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("setup_data")

REPO = "TruongVox/Cawot-dataset"
REPO_TYPE = "dataset"


def parse_shards(args) -> list[int]:
    if args.shards:
        return sorted(set(int(s.strip()) for s in args.shards.split(",")))
    return list(range(args.num_shards))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default="./pab_data")
    ap.add_argument("--shards", type=str, default=None,
                    help="Comma-separated shard IDs, e.g. '0,1,2,3,4'. Overrides --num-shards.")
    ap.add_argument("--num-shards", type=int, default=5,
                    help="Number of shards (0..N-1) if --shards not given. Default 5.")
    ap.add_argument("--no-cleanup", action="store_true",
                    help="Keep zip files after extraction.")
    ap.add_argument("--token", type=str, default=None,
                    help="HF token (only needed if the repo is gated).")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    output = Path(args.output)
    images_dir = output / "images"
    anns_dir = output / "annotations"
    cache_dir = output / ".hf_cache"
    images_dir.mkdir(parents=True, exist_ok=True)
    anns_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    shards = parse_shards(args)
    logger.info(f"shards to fetch: {shards}")

    for i in shards:
        # --- annotation file ---
        target_ann = anns_dir / f"imgs_{i}.json"
        if not target_ann.exists():
            try:
                src = hf_hub_download(
                    repo_id=REPO, filename=f"train/imgs_{i}.json",
                    repo_type=REPO_TYPE, cache_dir=str(cache_dir), token=args.token,
                )
                shutil.copy2(src, target_ann)
                logger.info(f"  ann imgs_{i}.json  ({target_ann.stat().st_size // 1024} KB)")
            except Exception as e:
                logger.error(f"  ✗ annotation imgs_{i}.json: {e}")
                continue
        else:
            logger.info(f"  ann imgs_{i}.json already present, skip")

        # --- image zip ---
        shard_dir = images_dir / f"imgs_{i}"
        if shard_dir.exists() and any(shard_dir.iterdir()):
            logger.info(f"  imgs_{i}/ already extracted, skip zip")
            continue
        try:
            zip_local = hf_hub_download(
                repo_id=REPO, filename=f"imgs_{i}.zip",
                repo_type=REPO_TYPE, cache_dir=str(cache_dir), token=args.token,
            )
            size_gb = Path(zip_local).stat().st_size / 1e9
            logger.info(f"  zip imgs_{i}.zip downloaded ({size_gb:.2f} GB), extracting...")
            with zipfile.ZipFile(zip_local) as zf:
                zf.extractall(images_dir)
            logger.info(f"  ✓ imgs_{i} extracted")
            if not args.no_cleanup:
                # Symlink target — delete the actual file in cache
                try:
                    real = Path(zip_local).resolve()
                    real.unlink()
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"  ✗ zip imgs_{i}.zip: {e}")

    # Final report
    extracted = sorted(d.name for d in images_dir.iterdir()
                       if d.is_dir() and d.name.startswith("imgs_"))
    anns = sorted(f.name for f in anns_dir.iterdir() if f.suffix == ".json")
    logger.info("")
    logger.info(f"Output: {output}")
    logger.info(f"  images/  : {len(extracted)} shards  {extracted[:8]}{'...' if len(extracted) > 8 else ''}")
    logger.info(f"  annotations/  : {len(anns)} files  {anns[:8]}{'...' if len(anns) > 8 else ''}")


if __name__ == "__main__":
    main()
