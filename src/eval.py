"""Retrieval evaluation.

Primary path (V0 sanity):
    Image-text retrieval on a held-out (image, caption) val split.
    Ground truth: caption[i] matches image[i] (1-to-1 by index).
    No person id required. This is the standard image-text retrieval setup
    used in CLIP / BLIP / ALBEF papers for cross-modal alignment.

Legacy path (for real PAB / gallery+query JSON):
    Person-id matching. Kept for compatibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import (
    FriendSubsetDataset,
    PABGalleryDataset,
    PABQueryDataset,
)
from .embed import load_clip
from .utils import ensure_dir, get_device, setup_logger

logger = setup_logger("eval")


# -----------------------------------------------------------------------------
# Image-text retrieval on held-out val (V0 PRIMARY)
# -----------------------------------------------------------------------------


@torch.no_grad()
def _encode_image_text_pairs(model, loader, device, amp: bool):
    """Encode each (image, caption) pair. Returns (img_feats, txt_feats) both
    L2-normalized, in dataloader order."""
    img_feats, txt_feats = [], []
    for batch in tqdm(loader, desc="encode val"):
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["text_tokens"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=amp):
            f_img = F.normalize(model.encode_image(images).float(), dim=-1)
            f_txt = F.normalize(model.encode_text(tokens).float(), dim=-1)
        img_feats.append(f_img.cpu().numpy())
        txt_feats.append(f_txt.cpu().numpy())
    return np.concatenate(img_feats, 0), np.concatenate(txt_feats, 0)


def _i2t_t2i_recall(sims: np.ndarray, k_values=(1, 5, 10)) -> dict:
    """sims: (N, N) cosine. sims[i, j] = sim(text_i, image_j).
    Diagonal is the correct pair. Returns text->image and image->text R@k.
    """
    n = sims.shape[0]
    out: dict[str, float] = {}

    # text -> image: for each row i, rank columns by descending sim, check if
    # column i appears in top-k
    t2i_ranks_desc = np.argsort(-sims, axis=1)
    correct_pos_t2i = np.argmax(t2i_ranks_desc == np.arange(n)[:, None], axis=1)
    for k in k_values:
        out[f"t2i_R@{k}"] = float((correct_pos_t2i < k).mean() * 100.0)

    # image -> text: argsort along rows of sims.T
    i2t_ranks_desc = np.argsort(-sims.T, axis=1)
    correct_pos_i2t = np.argmax(i2t_ranks_desc == np.arange(n)[:, None], axis=1)
    for k in k_values:
        out[f"i2t_R@{k}"] = float((correct_pos_i2t < k).mean() * 100.0)

    out["mean_R@1"] = 0.5 * (out["t2i_R@1"] + out["i2t_R@1"])
    out["n_pairs"] = int(n)
    return out


def evaluate_val_split(
    cfg: dict,
    manifest: pd.DataFrame,
    val_indices: np.ndarray,
    checkpoint_path: str | None,
    run_name: str,
) -> dict:
    """V0 sanity eval: image-text retrieval on held-out (image, caption) pairs."""
    device = get_device("cuda")
    out_dir = ensure_dir(cfg["eval"]["output_dir"])

    model, preprocess, tokenizer = load_clip(
        cfg["train"]["model"], cfg["train"]["pretrained"], device
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        state = state.get("model", state)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        logger.info("evaluating zero-shot CLIP (no checkpoint)")
    model.eval()

    val_ds = FriendSubsetDataset(
        manifest=manifest,
        image_transform=preprocess,
        tokenizer=tokenizer,
        indices=val_indices,
    )
    loader = DataLoader(
        val_ds,
        batch_size=cfg["eval"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )

    img_feats, txt_feats = _encode_image_text_pairs(model, loader, device, cfg["train"]["amp"])
    sims = txt_feats @ img_feats.T   # (N, N)  text→image
    metrics = _i2t_t2i_recall(sims, k_values=tuple(cfg["eval"]["k_values"]))
    logger.info(f"[{run_name}] {metrics}")

    out_json = Path(out_dir) / f"{run_name}_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# -----------------------------------------------------------------------------
# Legacy: person-id retrieval on real gallery+queries JSON
# -----------------------------------------------------------------------------


@torch.no_grad()
def _encode_gallery(model, loader, device, amp: bool):
    feats, pids = [], []
    for batch in tqdm(loader, desc="encode gallery"):
        with torch.cuda.amp.autocast(enabled=amp):
            f = F.normalize(model.encode_image(batch["image"].to(device, non_blocking=True)).float(), dim=-1)
        feats.append(f.cpu().numpy())
        pids.append(batch["pid"].numpy())
    return np.concatenate(feats, 0), np.concatenate(pids, 0)


@torch.no_grad()
def _encode_queries(model, loader, device, amp: bool):
    feats, pids = [], []
    for batch in tqdm(loader, desc="encode queries"):
        with torch.cuda.amp.autocast(enabled=amp):
            f = F.normalize(model.encode_text(batch["text_tokens"].to(device, non_blocking=True)).float(), dim=-1)
        feats.append(f.cpu().numpy())
        pids.append(batch["pid"].numpy())
    return np.concatenate(feats, 0), np.concatenate(pids, 0)


def compute_metrics(sims: np.ndarray, q_pids: np.ndarray, g_pids: np.ndarray,
                    k_values=(1, 5, 10)) -> dict:
    Q, G = sims.shape
    sorted_idx = np.argsort(-sims, axis=1)
    sorted_pids = g_pids[sorted_idx]
    matches = (sorted_pids == q_pids[:, None]).astype(np.float32)
    out: dict = {}
    for k in k_values:
        out[f"R@{k}"] = float((matches[:, :k].sum(1) > 0).mean() * 100.0)
    ap_per_q = []
    for q in range(Q):
        m = matches[q]
        if m.sum() == 0:
            continue
        ranks = np.arange(1, G + 1)
        ap = ((np.cumsum(m) / ranks) * m).sum() / m.sum()
        ap_per_q.append(ap)
    out["mAP"] = float(np.mean(ap_per_q) * 100.0) if ap_per_q else 0.0
    out["num_queries"] = int(Q)
    out["num_gallery"] = int(G)
    return out


def evaluate(cfg: dict, checkpoint_path: str, run_name: str) -> dict:
    device = get_device("cuda")
    out_dir = ensure_dir(cfg["eval"]["output_dir"])
    model, preprocess, tokenizer = load_clip(
        cfg["train"]["model"], cfg["train"]["pretrained"], device
    )
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device)
        state = ckpt.get("model", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(f"loaded {checkpoint_path}  missing={len(missing)}  unexpected={len(unexpected)}")
    else:
        logger.info("evaluating zero-shot CLIP")
    model.eval()

    root = Path(cfg["data"]["root"])
    gallery_ds = PABGalleryDataset(root / cfg["data"]["test_gallery_json"], root, preprocess)
    query_ds = PABQueryDataset(root / cfg["data"]["test_query_json"], tokenizer)
    g_loader = DataLoader(gallery_ds, batch_size=cfg["eval"]["batch_size"],
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    q_loader = DataLoader(query_ds, batch_size=cfg["eval"]["batch_size"],
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    amp = cfg["train"]["amp"]
    g_feat, g_pid = _encode_gallery(model, g_loader, device, amp)
    q_feat, q_pid = _encode_queries(model, q_loader, device, amp)
    sims = q_feat @ g_feat.T
    metrics = compute_metrics(sims, q_pid, g_pid, k_values=tuple(cfg["eval"]["k_values"]))
    logger.info(f"[{run_name}] {metrics}")
    with open(Path(out_dir) / f"{run_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics
