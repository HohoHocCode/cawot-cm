"""PAB / CMP dataset loaders.

Expected JSON format (list of dicts):
  - train.json: [{"image_path": ..., "captions": [...], "id": int}, ...]
  - test_query.json: [{"caption": str, "id": int}, ...]  or {"captions":[str]}
  - test_gallery.json: [{"image_path": ..., "id": int}, ...]

If your PAB dump uses different keys, adjust _extract_caption / _extract_image_path.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


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


def _extract_image_path(entry: dict, root: Path) -> Path:
    for k in ("image_path", "image", "img_path", "file_path"):
        if k in entry:
            p = Path(entry[k])
            return p if p.is_absolute() else root / p
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
    """Synthetic (image, text) pairs for training / embedding extraction."""

    def __init__(
        self,
        json_path: str | Path,
        image_root: str | Path,
        image_transform: Callable,
        tokenizer: Callable | None = None,
        subset_size: int | None = None,
        indices: np.ndarray | None = None,
        seed: int = 42,
        return_index: bool = False,
    ):
        self.image_root = Path(image_root)
        self.transform = image_transform
        self.tokenizer = tokenizer
        self.return_index = return_index

        anns = load_annotations(json_path, subset_size=subset_size, seed=seed)
        if indices is not None:
            anns = [anns[i] for i in indices.tolist()]
        self.anns = anns

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int):
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
    """Real gallery images for evaluation."""

    def __init__(self, json_path: str | Path, image_root: str | Path, image_transform: Callable):
        self.image_root = Path(image_root)
        self.transform = image_transform
        self.anns = load_annotations(json_path)

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int):
        entry = self.anns[idx]
        img_path = _extract_image_path(entry, self.image_root)
        image = self.transform(_open_image(str(img_path)))
        pid = entry.get("id", entry.get("pid", idx))
        return {"image": image, "pid": int(pid), "index": idx}


class PABQueryDataset(Dataset):
    """Real text queries for evaluation."""

    def __init__(self, json_path: str | Path, tokenizer: Callable):
        self.tokenizer = tokenizer
        self.anns = load_annotations(json_path)

    def __len__(self) -> int:
        return len(self.anns)

    def __getitem__(self, idx: int):
        entry = self.anns[idx]
        caption = _extract_caption(entry)
        pid = entry.get("id", entry.get("pid", idx))
        return {
            "text_tokens": self.tokenizer([caption])[0],
            "caption": caption,
            "pid": int(pid),
            "index": idx,
        }
