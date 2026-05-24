"""CLIP embedding extraction for the train pool.

Output:
  - image_embeddings.npy  (N, D) float32, L2-normalized
  - text_embeddings.npy   (N, D) float32, L2-normalized (optional)
  - indices.npy           (N,) int64 — index into the train annotation list
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
        subset_size=cfg["data"]["subset_size"],
        seed=cfg["seed"],
        return_index=True,
    )
    N = len(dataset)
    logger.info(f"Pool size: {N}")

    loader = DataLoader(
        dataset,
        batch_size=cfg["embed"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    # Discover embed dim with a dry run
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
        img_buf[cursor : cursor + bsz] = f_img.cpu().numpy()

        if txt_buf is not None:
            tokens = batch["text_tokens"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                f_txt = model.encode_text(tokens)
                f_txt = torch.nn.functional.normalize(f_txt.float(), dim=-1)
            txt_buf[cursor : cursor + bsz] = f_txt.cpu().numpy()

        idx_buf[cursor : cursor + bsz] = batch["index"].numpy()
        cursor += bsz

    img_path = out_dir / "image_embeddings.npy"
    idx_path = out_dir / "indices.npy"
    np.save(img_path, img_buf)
    np.save(idx_path, idx_buf)
    paths = {"image": str(img_path), "indices": str(idx_path)}
    logger.info(f"Saved {img_path}  shape={img_buf.shape}")

    if txt_buf is not None:
        txt_path = out_dir / "text_embeddings.npy"
        np.save(txt_path, txt_buf)
        paths["text"] = str(txt_path)
        logger.info(f"Saved {txt_path}  shape={txt_buf.shape}")

    return paths
