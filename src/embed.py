"""CLIP embedding extraction for V0.

V0 path:  extract_image_embeddings(dataset, model, device, cfg)
    Takes any Dataset that yields {"image": tensor} and returns L2-normalized
    image embeddings (N, D) as a numpy float32 array, in dataset order.

Legacy path: extract_embeddings(cfg) — for the dummy/JSON sanity flow.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import PABTrainDataset
from .utils import ensure_dir, get_device, setup_logger

logger = setup_logger("embed")


def load_clip(model_name: str, pretrained: str, device: torch.device):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


# -----------------------------------------------------------------------------
# V0 PRIMARY
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_image_embeddings(
    dataset,
    model,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    amp: bool = True,
) -> np.ndarray:
    """Run `model.encode_image` over the dataset. L2-normalize.

    Returns (N, D) float32 array in dataset order.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    feats: list[np.ndarray] = []
    for batch in tqdm(loader, desc="extract image embeddings"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            f = model.encode_image(images)
            f = torch.nn.functional.normalize(f.float(), dim=-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


@torch.no_grad()
def extract_image_text_embeddings(
    dataset,
    model,
    device: torch.device,
    batch_size: int = 128,
    num_workers: int = 2,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """One pass over the dataset, encoding BOTH image and text. L2-normalized.

    Loads each image once and encodes its caption too — used by V1 which needs
    text embeddings for the cross-modal cost. `dataset` must yield
    {"image": tensor, "text_tokens": LongTensor}.

    Returns (image_embeddings, text_embeddings), each (N, D) float32.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    img_feats: list[np.ndarray] = []
    txt_feats: list[np.ndarray] = []
    for batch in tqdm(loader, desc="extract image+text embeddings"):
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["text_tokens"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            fi = torch.nn.functional.normalize(model.encode_image(images).float(), dim=-1)
            ft = torch.nn.functional.normalize(model.encode_text(tokens).float(), dim=-1)
        img_feats.append(fi.cpu().numpy())
        txt_feats.append(ft.cpu().numpy())
    return (np.concatenate(img_feats, 0).astype(np.float32),
            np.concatenate(txt_feats, 0).astype(np.float32))


# -----------------------------------------------------------------------------
# LEGACY (dummy / JSON path)
# -----------------------------------------------------------------------------


@torch.no_grad()
def extract_embeddings(cfg: dict) -> dict:
    device = get_device(cfg["embed"]["device"])
    out_dir = ensure_dir(cfg["embed"]["output_dir"])
    model, preprocess, tokenizer = load_clip(
        cfg["embed"]["model"], cfg["embed"]["pretrained"], device
    )
    dataset = PABTrainDataset(
        json_path=Path(cfg["data"]["root"]) / cfg["data"]["train_json"],
        image_root=cfg["data"]["root"],
        image_transform=preprocess,
        tokenizer=tokenizer if cfg["embed"]["compute_text"] else None,
        subset_size=cfg["data"].get("subset_size"),
        seed=cfg["seed"],
        return_index=True,
    )
    N = len(dataset)
    logger.info(f"Pool size: {N}")
    loader = DataLoader(
        dataset, batch_size=cfg["embed"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], pin_memory=True,
    )

    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        sample = next(iter(loader))
        img_feat = model.encode_image(sample["image"].to(device))
        D = img_feat.shape[1]

    img_buf = np.zeros((N, D), dtype=np.float32)
    txt_buf = np.zeros((N, D), dtype=np.float32) if cfg["embed"]["compute_text"] else None
    idx_buf = np.zeros((N,), dtype=np.int64)
    cursor = 0
    for batch in tqdm(loader, desc="Embedding"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            f_img = model.encode_image(images)
            f_img = torch.nn.functional.normalize(f_img.float(), dim=-1)
        bsz = f_img.shape[0]
        img_buf[cursor:cursor + bsz] = f_img.cpu().numpy()
        if txt_buf is not None:
            tokens = batch["text_tokens"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                f_txt = model.encode_text(tokens)
                f_txt = torch.nn.functional.normalize(f_txt.float(), dim=-1)
            txt_buf[cursor:cursor + bsz] = f_txt.cpu().numpy()
        idx_buf[cursor:cursor + bsz] = batch["index"].numpy()
        cursor += bsz

    img_path = out_dir / "image_embeddings.npy"
    np.save(img_path, img_buf)
    np.save(out_dir / "indices.npy", idx_buf)
    paths = {"image": str(img_path), "indices": str(out_dir / "indices.npy")}
    if txt_buf is not None:
        np.save(out_dir / "text_embeddings.npy", txt_buf)
        paths["text"] = str(out_dir / "text_embeddings.npy")
    return paths
