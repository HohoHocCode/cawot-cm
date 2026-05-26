"""Dataset loaders for the friend's qproxy precomputed subset.

The friend's `image_subset_manifest.parquet` has these columns:
  row_id, image_id, annotation_file, annotation_line, annotation_image,
  shard, action_dir, image_stem, image_path, caption, scene, label_type,
  action, missing_image

`image_path` is an absolute path baked at friend's Kaggle run, e.g.
  /kaggle/input/datasets/vnhtbo/pab-eccv26-track4-train-webp-part-01-05/Part 1/imgs_0/imgs_0/goal/0.webp

To run elsewhere (different Kaggle dataset slug, Colab, local), set
`path_remap: {old_prefix: new_prefix}` in the config — paths are rewritten
on load. Webp images are read transparently by PIL.

`image_subset_embeddings.npy` is parallel to the manifest's row order:
row i of manifest ↔ row i of embeddings.

Legacy JSON-based loaders are kept as PABTrainDataset/PABGalleryDataset/
PABQueryDataset for sanity tests and dummy data.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# Friend's qproxy-based loaders (PRIMARY path for V0)
# -----------------------------------------------------------------------------


def _remap_path(path: str, remap: dict[str, str] | None) -> str:
    if not remap:
        return path
    for old, new in remap.items():
        if path.startswith(old):
            return new + path[len(old):]
    return path


def load_manifest(
    manifest_path: str | Path,
    path_remap: dict[str, str] | None = None,
    drop_missing: bool = True,
) -> pd.DataFrame:
    """Load the friend's image_subset_manifest.parquet, optionally remap paths.

    Returns a DataFrame with stable integer index 0..N-1 (parallel to the
    embedding npy). Original `row_id` (which is non-contiguous after filtering)
    is preserved as a column.
    """
    df = pd.read_parquet(manifest_path)
    if path_remap:
        df["image_path"] = df["image_path"].map(lambda p: _remap_path(p, path_remap))
    if drop_missing and "missing_image" in df.columns:
        df = df[~df["missing_image"]].copy()
    df = df.reset_index(drop=True)
    return df


def load_embeddings(npy_path: str | Path, expected_n: int | None = None) -> np.ndarray:
    arr = np.load(npy_path)
    if expected_n is not None and arr.shape[0] != expected_n:
        raise ValueError(
            f"Embedding count {arr.shape[0]} != manifest rows {expected_n}. "
            f"Manifest and embeddings are out of sync."
        )
    # Normalize to unit length (clustering uses spherical / cosine)
    norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
    return (arr / norms).astype(np.float32)


class FriendSubsetDataset(Dataset):
    """(image, caption) pairs sourced from the friend's manifest.

    Each item:
        {"image": tensor, "caption": str, "text_tokens": LongTensor (optional),
         "row_idx": int}

    `row_idx` is the dataset position (0..len-1), which matches the embedding
    npy ordering. Use it to look up corresponding embedding.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_transform: Callable,
        tokenizer: Callable | None = None,
        indices: np.ndarray | None = None,
    ):
        if indices is not None:
            manifest = manifest.iloc[indices].reset_index(drop=True)
        self.manifest = manifest
        self.transform = image_transform
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int):
        row = self.manifest.iloc[idx]
        img = Image.open(row["image_path"])
        if img.mode != "RGB":
            img = img.convert("RGB")
        image = self.transform(img)
        caption = str(row["caption"])
        sample: dict = {
            "image": image,
            "caption": caption,
            "row_idx": int(idx),
        }
        if self.tokenizer is not None:
            sample["text_tokens"] = self.tokenizer([caption])[0]
        return sample


def make_train_val_split(
    n: int, val_size: int, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Random train/val split over manifest rows.

    Used by V0 to carve out a held-out (image, caption) val set for
    image-text retrieval R@k (sanity metric — no person id required).
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    val_idx = np.sort(perm[:val_size])
    train_idx = np.sort(perm[val_size:])
    return train_idx, val_idx


# -----------------------------------------------------------------------------
# Legacy JSON-based loaders (kept for sanity / dummy data path)
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
