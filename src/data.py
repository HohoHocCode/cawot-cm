"""Dataset loaders for V0.

PRIMARY path (V0):
    TrainPoolDataset — reads PAB JSONL annotations (one entry per line) +
    resolves images on the local filesystem. Independent of friend's qproxy
    work. V0 samples its own random subset of the full train pool, extracts
    its own CLIP-B/16 embeddings, clusters, selects, trains.

Each JSONL line has the schema (from friend's `imgs_N.json`):
    {"image": "train/imgs_0/goal/0.jpg",
     "caption": "...",
     "image_id": "0_0",
     "scene": "...",
     "normal": "..."}

`image` is the path AS LOGGED IN ANNOTATIONS (extension `.jpg`).
On disk friend's images are `.webp` and live under a Part-N structure:
    <image_root>/Part 1/imgs_0/imgs_0/goal/0.webp
We discover the Part for each shard at __init__ and rewrite paths.

LEGACY loaders (PABTrainDataset, PABGalleryDataset, PABQueryDataset) kept
for dummy-data sanity tests.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# V0 PRIMARY: TrainPoolDataset
# -----------------------------------------------------------------------------


def _discover_shards(image_root: Path) -> dict[str, Path]:
    """Map shard name ('imgs_0') → directory containing its action subfolders.

    Friend's Kaggle layout: <image_root>/Part X/imgs_N/imgs_N/<action_dir>/Y.webp
    Also accepts flat: <image_root>/imgs_N/<action_dir>/Y.webp
    """
    roots: dict[str, Path] = {}
    for child in sorted(image_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name.lower().startswith("part"):
            for shard_dir in sorted(child.iterdir()):
                if shard_dir.is_dir() and shard_dir.name.startswith("imgs_"):
                    nested = shard_dir / shard_dir.name
                    roots[shard_dir.name] = nested if nested.is_dir() else shard_dir
        elif child.name.startswith("imgs_"):
            roots[child.name] = child
    return roots


def _iter_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def build_pool(
    annotations_dir: str | Path,
    image_root: str | Path,
    sample_size: int | None = None,
    seed: int = 42,
) -> tuple[list[dict], dict[str, Path]]:
    """Load JSONL annotations, filter to shards present on disk, then
    optionally random-sample down to `sample_size`.

    Returns (annotations list, shard_roots map).
    """
    image_root = Path(image_root)
    annotations_dir = Path(annotations_dir)

    shard_roots = _discover_shards(image_root)
    if not shard_roots:
        raise RuntimeError(
            f"No shard folders (imgs_N) found under {image_root}. "
            "Check the Kaggle image dataset is mounted correctly."
        )

    anns: list[dict] = []
    for jsonl_file in sorted(annotations_dir.glob("imgs_*.json")):
        shard_name = jsonl_file.stem
        if shard_name not in shard_roots:
            continue
        for e in _iter_jsonl(jsonl_file):
            if "image" not in e or "caption" not in e:
                continue
            anns.append({
                "image": e["image"],
                "caption": e["caption"],
                "image_id": e.get("image_id", ""),
            })

    if not anns:
        raise RuntimeError(
            f"Loaded 0 annotations from {annotations_dir}. "
            "Check the JSONL files exist and match shard names on disk."
        )

    if sample_size is not None and sample_size < len(anns):
        rng = np.random.RandomState(seed)
        chosen = rng.choice(len(anns), size=sample_size, replace=False)
        anns = [anns[i] for i in sorted(chosen.tolist())]

    return anns, shard_roots


def make_train_val_split(n: int, val_size: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Random train/val split over pool indices.

    val_idx is held out for image-text retrieval R@k eval.
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:val_size])
    train_idx = np.sort(perm[val_size:])
    return train_idx, val_idx


class TrainPoolDataset(Dataset):
    """V0 dataset: (image, caption) pairs from JSONL annotations + on-disk images."""

    def __init__(
        self,
        anns: list[dict],
        shard_roots: dict[str, Path],
        image_transform: Callable,
        tokenizer: Callable | None = None,
        indices: np.ndarray | None = None,
        return_index: bool = False,
    ):
        if indices is not None:
            anns = [anns[i] for i in indices.tolist()]
        self.anns = anns
        self.shard_roots = shard_roots
        self.transform = image_transform
        self.tokenizer = tokenizer
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.anns)

    def _resolve_image_path(self, image_field: str) -> Path:
        # image_field like "train/imgs_0/goal/0.jpg"
        rel = image_field
        if rel.startswith("train/"):
            rel = rel[len("train/"):]
        rel_path = Path(rel)
        shard = rel_path.parts[0]
        if shard not in self.shard_roots:
            raise KeyError(f"Shard {shard} not found on disk. Available: {list(self.shard_roots)[:5]}...")
        sub = Path(*rel_path.parts[1:]).with_suffix(".webp")
        return self.shard_roots[shard] / sub

    def __getitem__(self, idx: int):
        entry = self.anns[idx]
        img_path = self._resolve_image_path(entry["image"])
        img = Image.open(img_path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        image = self.transform(img)
        caption = entry["caption"]
        sample: dict = {"image": image, "caption": caption}
        if self.tokenizer is not None:
            sample["text_tokens"] = self.tokenizer([caption])[0]
        if self.return_index:
            sample["index"] = idx
        return sample


# -----------------------------------------------------------------------------
# LEGACY: JSON-based loaders for dummy-data sanity path
# -----------------------------------------------------------------------------


def _open_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def _extract_caption(entry: dict) -> str:
    if "caption" in entry:
        return entry["caption"]
    if "captions" in entry and entry["captions"]:
        caps = entry["captions"]
        return random.choice(caps) if isinstance(caps, list) else caps
    raise KeyError(f"No caption field in entry keys: {list(entry.keys())}")


def _extract_image_path(entry: dict, root: Path, images_subdir: str = "images") -> Path:
    for k in ("image_path", "image", "img_path", "file_path"):
        if k not in entry:
            continue
        raw = entry[k]
        p = Path(raw)
        if p.is_absolute():
            return p
        cand1 = root / p
        if cand1.exists():
            return cand1
        cand2 = root / images_subdir / p.name
        if cand2.exists():
            return cand2
        return cand2
    raise KeyError(f"No image-path field in entry keys: {list(entry.keys())}")


def load_annotations(json_path: str | Path, subset_size: int | None = None, seed: int = 42) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        anns = json.load(f)
    if isinstance(anns, dict) and "data" in anns:
        anns = anns["data"]
    if subset_size is not None and subset_size < len(anns):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(anns), size=subset_size, replace=False)
        anns = [anns[i] for i in sorted(idx.tolist())]
    return anns


class PABTrainDataset(Dataset):
    def __init__(self, json_path, image_root, image_transform, tokenizer=None,
                 subset_size=None, indices=None, seed=42, return_index=False):
        self.image_root = Path(image_root)
        self.transform = image_transform
        self.tokenizer = tokenizer
        self.return_index = return_index
        anns = load_annotations(json_path, subset_size=subset_size, seed=seed)
        if indices is not None:
            anns = [anns[i] for i in indices.tolist()]
        self.anns = anns

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        entry = self.anns[idx]
        img_path = _extract_image_path(entry, self.image_root)
        image = self.transform(_open_image(str(img_path)))
        caption = _extract_caption(entry)
        sample: dict = {"image": image, "caption": caption}
        if self.tokenizer is not None:
            sample["text_tokens"] = self.tokenizer([caption])[0]
        if self.return_index:
            sample["index"] = idx
        return sample


class PABGalleryDataset(Dataset):
    def __init__(self, json_path, image_root, image_transform):
        self.image_root = Path(image_root)
        self.transform = image_transform
        self.anns = load_annotations(json_path)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        entry = self.anns[idx]
        img_path = _extract_image_path(entry, self.image_root)
        image = self.transform(_open_image(str(img_path)))
        pid = entry.get("id", entry.get("pid", idx))
        return {"image": image, "pid": int(pid), "index": idx}


class PABQueryDataset(Dataset):
    def __init__(self, json_path, tokenizer):
        self.tokenizer = tokenizer
        self.anns = load_annotations(json_path)

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        entry = self.anns[idx]
        caption = _extract_caption(entry)
        pid = entry.get("id", entry.get("pid", idx))
        return {
            "text_tokens": self.tokenizer([caption])[0],
            "caption": caption,
            "pid": int(pid),
            "index": idx,
        }
