"""Generate tiny synthetic PAB-like data for pipeline sanity check.

Creates random RGB images + fake captions in PAB-compatible JSON.
Use this to verify the pipeline runs end-to-end before plugging in real PAB.

Usage:
  python scripts/make_dummy_data.py --root ./dummy_pab --n-train 2000 --n-query 100 --n-gallery 500
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


CAPTION_TEMPLATES = [
    "A person wearing {color} {top} and {bottom_color} {bottom} is {action}",
    "Someone in {color} clothing {action} on the street",
    "A {age} person with {hair} hair walking {direction}",
    "A person carrying a {bag} bag {action}",
    "Someone wearing {color} {top} stands near the {location}",
]
COLORS = ["red", "blue", "green", "black", "white", "yellow", "gray"]
TOPS = ["shirt", "jacket", "t-shirt", "hoodie", "sweater"]
BOTTOMS = ["jeans", "trousers", "shorts", "skirt"]
ACTIONS = ["walking", "running", "standing", "falling", "sitting", "talking"]
AGES = ["young", "elderly", "middle-aged"]
HAIRS = ["short", "long", "curly", "straight"]
DIRECTIONS = ["left", "right", "forward", "backward"]
BAGS = ["red", "black", "blue"]
LOCATIONS = ["entrance", "corner", "doorway", "stairs"]


def make_caption(rng: random.Random) -> str:
    template = rng.choice(CAPTION_TEMPLATES)
    return template.format(
        color=rng.choice(COLORS),
        top=rng.choice(TOPS),
        bottom_color=rng.choice(COLORS),
        bottom=rng.choice(BOTTOMS),
        action=rng.choice(ACTIONS),
        age=rng.choice(AGES),
        hair=rng.choice(HAIRS),
        direction=rng.choice(DIRECTIONS),
        bag=rng.choice(BAGS),
        location=rng.choice(LOCATIONS),
    )


def make_image(path: Path, seed: int) -> None:
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 255, size=(64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="./dummy_pab")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-query", type=int, default=100)
    ap.add_argument("--n-gallery", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "images" / "gallery").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # n_ids: each person id gets multiple images/captions
    n_ids = max(20, args.n_query)

    # Train
    train = []
    for i in tqdm(range(args.n_train), desc="train"):
        pid = rng.randint(0, n_ids - 1)
        img_rel = f"images/train/{i:06d}.jpg"
        make_image(root / img_rel, seed=i)
        train.append({"image_path": img_rel, "captions": [make_caption(rng)], "id": pid})
    with open(root / "train.json", "w") as f:
        json.dump(train, f)

    # Gallery
    gallery = []
    for i in tqdm(range(args.n_gallery), desc="gallery"):
        pid = rng.randint(0, n_ids - 1)
        img_rel = f"images/gallery/{i:06d}.jpg"
        make_image(root / img_rel, seed=10_000 + i)
        gallery.append({"image_path": img_rel, "id": pid})
    with open(root / "test_gallery.json", "w") as f:
        json.dump(gallery, f)

    # Query
    queries = []
    for i in tqdm(range(args.n_query), desc="query"):
        pid = i % n_ids
        queries.append({"caption": make_caption(rng), "id": pid})
    with open(root / "test_query.json", "w") as f:
        json.dump(queries, f)

    print(f"\nDummy data written to {root}")
    print(f"  train: {len(train)}, gallery: {len(gallery)}, queries: {len(queries)}, "
          f"n_ids: {n_ids}")


if __name__ == "__main__":
    main()
