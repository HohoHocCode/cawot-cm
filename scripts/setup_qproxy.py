"""Download Q_proxy (queries.json + companion files) from a shared Google Drive
folder, then verify the queries.json is present.

The qproxy/ folder on Drive (default ID below) contains:
  queries.json                          ← we use this (2,593 LLM queries)
  image_subset_embeddings.npy           ← NOT used by V2 (EVA02 1024-d)
  image_subset_manifest.parquet         ← NOT used
  text_embeddings.npy                   ← NOT used (EVA02; we re-encode w/ CLIP)
  max_sim_raw.npy                       ← NOT used
  filter_metadata.json                  ← informational only

V2 needs only queries.json; we re-encode its captions with our CLIP-B/16 text
encoder so embeddings live in the SAME space as the train pool.

Usage:
  python scripts/setup_qproxy.py --output ./qproxy
  python scripts/setup_qproxy.py --output ./qproxy --only-queries  # skip big files
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import setup_logger

logger = setup_logger("setup_qproxy")

# The user's shared qproxy folder
DEFAULT_FOLDER_ID = "1SumS21mA_YS-sykOD_smB8DUO1famPju"


def find_queries_json(root: Path) -> Path | None:
    """Locate queries.json under `root` (possibly nested)."""
    if (root / "queries.json").exists():
        return root / "queries.json"
    cands = list(root.rglob("queries.json"))
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="./qproxy",
                    help="Output folder (Drive contents land here).")
    ap.add_argument("--folder-id", default=DEFAULT_FOLDER_ID,
                    help="Google Drive folder ID. Default = the user's shared qproxy.")
    ap.add_argument("--only-queries", action="store_true",
                    help="Only keep queries.json; delete the other (large) files after download.")
    args = ap.parse_args()

    import gdown

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    logger.info(f"downloading folder {args.folder_id} → {output}")
    try:
        gdown.download_folder(url, output=str(output), quiet=False, use_cookies=False)
    except Exception as e:
        logger.error(f"gdown failed: {e}")
        logger.error("Make sure the Drive folder is shared (anyone with link, viewer).")
        return 1

    q = find_queries_json(output)
    if q is None:
        logger.error(f"queries.json NOT FOUND under {output}. Listing:")
        for p in output.rglob("*"):
            logger.error(f"  {p.relative_to(output)}")
        return 1
    logger.info(f"✓ queries.json at: {q}")

    if args.only_queries:
        # Move queries.json to output root, delete everything else.
        target = output / "queries.json"
        if q != target:
            target.write_bytes(q.read_bytes())
        for p in output.rglob("*"):
            if p.is_file() and p.name != "queries.json":
                try:
                    p.unlink()
                except Exception:
                    pass
        # Clean empty dirs
        for p in sorted(output.rglob("*"), key=lambda x: -len(str(x))):
            if p.is_dir() and p != output:
                try:
                    p.rmdir()
                except OSError:
                    pass
        logger.info(f"--only-queries: kept only {target}")

    logger.info("\nDone. To use in V2, set in config.yaml:")
    logger.info(f"  qproxy.queries_json_path: {q}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
