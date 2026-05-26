"""Download + extract the Cawot PAB dataset from Hugging Face.

Source: https://huggingface.co/datasets/TruongVox/Cawot-dataset

Files on the HF repo:
  - imgs_0.zip  ...  imgs_74.zip   (75 archives, ~1.4 GB each, ~104 GB total)
  - train.json, test_query.json, test_gallery.json    (annotation files; user uploads separately)

Final on-disk layout produced by this script:

  pab_data/
  ├── images/                 # ALL synthetic + gallery images flat
  │   ├── 000000.jpg
  │   ├── 000001.jpg
  │   └── ...
  ├── train.json
  ├── test_query.json
  └── test_gallery.json

Usage:
  # Full download (~104 GB)
  python scripts/download_pab.py --root ./pab_data

  # Subset for Colab / testing (first N zips only)
  python scripts/download_pab.py --root ./pab_data --num-zips 3

  # Skip extraction (download only)
  python scripts/download_pab.py --root ./pab_data --no-extract

The script is idempotent — already-downloaded zips and extracted images are skipped.
After --num-zips runs, also run scripts/filter_annotations.py if you want train.json
trimmed to only entries whose images are present locally.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("download_pab")

REPO_ID = "TruongVox/Cawot-dataset"
REPO_TYPE = "dataset"
NUM_ZIPS_TOTAL = 75
ANNOTATION_FILES = ["train.json", "test_query.json", "test_gallery.json"]


def hf_download(filename: str, local_dir: Path, token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type=REPO_TYPE,
        local_dir=str(local_dir),
        token=token,
    )
    return Path(path)


def extract_zip_flat(zip_path: Path, target_dir: Path) -> int:
    """Extract a zip flat into target_dir (ignores any internal folder structure).

    Returns number of files extracted (new files only — existing files are skipped).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            # Flatten: take only the basename
            name = Path(member.filename).name
            if not name:
                continue
            out_path = target_dir / name
            if out_path.exists():
                continue
            with zf.open(member) as src, open(out_path, "wb") as dst:
                dst.write(src.read())
            n_new += 1
    return n_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./pab_data",
                    help="Output root. Images go to <root>/images/, JSONs to <root>/")
    ap.add_argument("--num-zips", type=int, default=NUM_ZIPS_TOTAL,
                    help=f"How many imgs_*.zip to download (1..{NUM_ZIPS_TOTAL}). "
                         "Use small number for Colab.")
    ap.add_argument("--zip-start", type=int, default=0,
                    help="Starting zip index (download imgs_{start}.zip ... imgs_{start+num-1}.zip)")
    ap.add_argument("--no-extract", action="store_true",
                    help="Download zips but skip extraction.")
    ap.add_argument("--delete-zips-after-extract", action="store_true",
                    help="Free disk by deleting zip files after successful extraction.")
    ap.add_argument("--skip-annotations", action="store_true",
                    help="Skip downloading the JSON annotation files (e.g. if not uploaded yet).")
    ap.add_argument("--token", type=str, default=None,
                    help="HF token (only needed if repo is gated/private).")
    args = ap.parse_args()

    root = Path(args.root)
    images_dir = root / "images"
    cache_dir = root / "_zips"
    root.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1) Annotation JSONs
    if not args.skip_annotations:
        logger.info("Downloading annotation JSON files...")
        for fname in ANNOTATION_FILES:
            try:
                p = hf_download(fname, root, token=args.token)
                logger.info(f"  ✓ {fname}  ({p.stat().st_size / 1024:.1f} KB)")
            except Exception as e:
                logger.warning(
                    f"  ⚠ Could not fetch {fname} from HF: {e}\n"
                    f"    If annotations are not yet on HF, place them manually at {root / fname}"
                )

    # 2) Image zips
    end = min(args.zip_start + args.num_zips, NUM_ZIPS_TOTAL)
    logger.info(f"Downloading imgs_{args.zip_start}.zip ... imgs_{end - 1}.zip "
                f"({end - args.zip_start} zips)")

    total_new = 0
    for i in range(args.zip_start, end):
        zip_name = f"imgs_{i}.zip"
        try:
            zip_path = hf_download(zip_name, cache_dir, token=args.token)
        except Exception as e:
            logger.error(f"  ✗ {zip_name} download failed: {e}")
            continue
        logger.info(f"  ✓ downloaded {zip_name}  ({zip_path.stat().st_size / 1e9:.2f} GB)")

        if args.no_extract:
            continue

        n_new = extract_zip_flat(zip_path, images_dir)
        total_new += n_new
        logger.info(f"  ✓ extracted {zip_name}  (+{n_new} new images)")

        if args.delete_zips_after_extract:
            zip_path.unlink()
            logger.info(f"  ✓ deleted {zip_name} to save disk")

    total_imgs = sum(1 for _ in images_dir.iterdir())
    logger.info(f"\nDone. {total_imgs} images now in {images_dir} (+{total_new} this run).")

    # 3) Summary of expected JSON files
    print("\nExpected files at root:")
    for fname in ANNOTATION_FILES:
        p = root / fname
        status = "✓" if p.exists() else "✗ MISSING"
        print(f"  {status}  {p}")


if __name__ == "__main__":
    main()
