"""Q_proxy loading + encoding for V2 budget allocation.

V2 uses a pool of LLM-generated synthetic text queries (Q_proxy) as a
distribution anchor: for each k-means cluster of the train pool, we compute
the Wasserstein distance between the cluster's TEXT embeddings and the
Q_proxy embeddings, then over-allocate budget to clusters far from Q_proxy
(under-covered by test-like queries).

The friend's qproxy/ folder on Drive has text_embeddings.npy in EVA02-E-14
(1024-dim) space. Our pool is encoded with CLIP ViT-B/16 (512-dim). To keep
everything in a single embedding space, we IGNORE friend's pre-computed
text embeddings and RE-ENCODE the 2,593 query captions with our CLIP-B/16
text encoder.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def load_qproxy_captions(json_path: str | Path) -> list[str]:
    """Load Q_proxy queries (JSON array OR JSONL). Returns caption strings.

    Friend's queries.json is JSONL with schema:
        {"query_index": "...", "caption": "...", "change": "..."}
    Robust to both JSONL (one object / line) and JSON-array.
    """
    p = Path(json_path)
    text = p.read_text(encoding="utf-8").strip()
    entries: list[dict] = []
    if text.startswith("["):
        entries = json.loads(text)
    else:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    captions: list[str] = []
    for e in entries:
        cap = e.get("caption", e.get("text", ""))
        if isinstance(cap, str) and cap.strip():
            captions.append(cap.strip())
    if not captions:
        raise RuntimeError(f"No captions found in {json_path} — check schema.")
    return captions


@torch.no_grad()
def encode_captions_clip(
    captions: Iterable[str],
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    amp: bool = True,
) -> np.ndarray:
    """Encode caption strings with `model.encode_text`. L2-normalize. Returns (N, D)."""
    captions = list(captions)
    feats: list[np.ndarray] = []
    for i in tqdm(range(0, len(captions), batch_size), desc="encode Q_proxy"):
        batch = captions[i : i + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
            f = model.encode_text(tokens)
            f = F.normalize(f.float(), dim=-1)
        feats.append(f.cpu().numpy())
    return np.concatenate(feats, 0).astype(np.float32)


def load_or_encode_qproxy(
    json_path: str | Path,
    cache_path: str | Path,
    model,
    tokenizer,
    device: torch.device,
    batch_size: int = 256,
    amp: bool = True,
) -> np.ndarray:
    """Cached: load encoded Q_proxy embeddings from `cache_path` if present,
    else read `json_path`, encode with CLIP text encoder, save to cache."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        emb = np.load(cache_path)
        return emb
    captions = load_qproxy_captions(json_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb = encode_captions_clip(captions, model, tokenizer, device, batch_size, amp)
    np.save(cache_path, emb)
    return emb
