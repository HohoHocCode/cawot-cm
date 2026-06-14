"""
cawot/data.py
=============
Embedding extraction (CLIP), caching, and proxy-family construction.

Two entry points:
  * embed_pairs(...)        -- encode (image, caption) pairs with CLIP, cache to
                               memory-mapped .npy so later stages never re-encode.
  * build_proxy_families... -- construct Q_1 (training-attribute templates),
                               Q_2 (paraphrases). Q_3 (validation-failure) is a
                               SEPARATE ablation helper and never part of main.

If CLIP / torch are unavailable (e.g. CPU smoke test), ``make_synthetic_dataset``
produces structured fake embeddings so the whole pipeline can be exercised.
"""
from __future__ import annotations

import os
import numpy as np


# --------------------------------------------------------------------------- #
#  CLIP embedding with caching                                                  #
# --------------------------------------------------------------------------- #
def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def embed_pairs(
    image_paths: list[str],
    captions: list[str],
    cache_dir: str,
    model_name: str = "ViT-B/16",
    batch_size: int = 256,
    device: str | None = None,
    overwrite: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode image-caption pairs with CLIP; cache z_v, z_t as memmapped .npy.

    Returns (z_v, z_t), both (N, d) float32 and L2-normalized.
    """
    os.makedirs(cache_dir, exist_ok=True)
    pv = os.path.join(cache_dir, "z_v.npy")
    pt = os.path.join(cache_dir, "z_t.npy")
    if (not overwrite) and os.path.exists(pv) and os.path.exists(pt):
        return np.load(pv, mmap_mode="r"), np.load(pt, mmap_mode="r")

    import torch
    import clip  # openai/CLIP
    from PIL import Image

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    N = len(captions)
    zv_list, zt_list = [], []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            j = min(N, i + batch_size)
            imgs = torch.stack(
                [preprocess(Image.open(p).convert("RGB"))
                 for p in image_paths[i:j]]).to(device)
            toks = clip.tokenize(captions[i:j], truncate=True).to(device)
            zv = model.encode_image(imgs).float().cpu().numpy()
            zt = model.encode_text(toks).float().cpu().numpy()
            zv_list.append(zv); zt_list.append(zt)
    z_v = _l2norm(np.concatenate(zv_list, axis=0)).astype(np.float32)
    z_t = _l2norm(np.concatenate(zt_list, axis=0)).astype(np.float32)
    np.save(pv, z_v); np.save(pt, z_t)
    return z_v, z_t


def embed_texts(texts: list[str], model_name: str = "ViT-B/16",
                device: str | None = None, batch_size: int = 256) -> np.ndarray:
    """Encode a list of query texts with CLIP text encoder (normalized)."""
    import torch
    import clip
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = clip.load(model_name, device=device)
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            toks = clip.tokenize(texts[i:i + batch_size], truncate=True).to(device)
            out.append(model.encode_text(toks).float().cpu().numpy())
    return _l2norm(np.concatenate(out, axis=0)).astype(np.float32)


# --------------------------------------------------------------------------- #
#  Proxy-family construction                                                    #
# --------------------------------------------------------------------------- #
ATTR_TEMPLATES = [
    "a person {a}",
    "a photo of a person {a}",
    "someone {a} in a public place",
    "a surveillance image of a person {a}",
]


def build_Q1_templates(attributes: list[str]) -> list[str]:
    """Q_1: templated training-attribute queries.

    ``attributes`` are training-side phrases (action / anomaly / scene) such as
    'falling down', 'lying on the ground', 'running'. Returns one query string
    per (attribute x template).
    """
    out = []
    for a in attributes:
        for tmpl in ATTR_TEMPLATES:
            out.append(tmpl.format(a=a))
    return out


def build_Q2_paraphrases(q1: list[str], paraphraser=None) -> list[str]:
    """Q_2: paraphrases of Q_1.

    If ``paraphraser`` (callable str->list[str]) is given it is used; otherwise a
    light rule-based expansion is applied so the pipeline runs without an LLM.
    """
    if paraphraser is not None:
        out = []
        for q in q1:
            out.extend(paraphraser(q))
        return out
    # rule-based fallback
    out = []
    for q in q1:
        out.append(q.replace("a person", "an individual"))
        out.append(q.replace("a photo of", "an image showing"))
    return out


def build_proxy_families(
    attributes: list[str],
    model_name: str = "ViT-B/16",
    device: str | None = None,
    paraphraser=None,
) -> tuple[list[np.ndarray], list[str]]:
    """Return (proxy_embeddings, names) for the MAIN families {Q_1, Q_2}.

    Q_3 (validation-failure) is intentionally excluded -- see
    ``build_Q3_validation_failures`` for the ablation-only helper.
    """
    q1 = build_Q1_templates(attributes)
    q2 = build_Q2_paraphrases(q1, paraphraser)
    e1 = embed_texts(q1, model_name, device)
    e2 = embed_texts(q2, model_name, device)
    return [e1, e2], ["Q1_templates", "Q2_paraphrases"]


def build_Q3_validation_failures(failure_captions: list[str],
                                 model_name: str = "ViT-B/16",
                                 device: str | None = None) -> np.ndarray:
    """ABLATION ONLY. Q_3 from validation-failure queries.

    Must be built strictly from a clean validation split -- never from test
    captions, identities, labels, or the test-time query distribution.
    """
    return embed_texts(failure_captions, model_name, device)


# --------------------------------------------------------------------------- #
#  Synthetic dataset for CPU smoke testing                                      #
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(
    n: int = 45000,
    d: int = 64,
    n_groups: int = 20,
    n_proxy: int = 2,
    proxy_queries: int = 200,
    noise_groups: int = 3,
    seed: int = 0,
):
    """Structured fake CLIP-like data on the unit sphere.

    Creates ``n_groups`` semantic clusters. A few groups are 'noisy' (high
    intra-variance) to test whether u_c spuriously inflates there. Proxy
    families concentrate on a subset of groups so relevance is non-uniform.

    Returns dict with z_v, z_t, proxy_emb (list), and metadata for eval.
    """
    rng = np.random.default_rng(seed)
    centers_v = _l2norm(rng.normal(size=(n_groups, d)))
    centers_t = _l2norm(rng.normal(size=(n_groups, d)))
    # group assignment (uneven sizes)
    probs = rng.dirichlet(np.ones(n_groups) * 2.0)
    groups = rng.choice(n_groups, size=n, p=probs)

    noisy = set(rng.choice(n_groups, size=noise_groups, replace=False).tolist())
    zv = np.zeros((n, d), np.float32)
    zt = np.zeros((n, d), np.float32)
    for i in range(n):
        g = groups[i]
        spread = 0.6 if g in noisy else 0.15
        zv[i] = centers_v[g] + rng.normal(scale=spread, size=d)
        zt[i] = centers_t[g] + rng.normal(scale=spread, size=d)
    zv = _l2norm(zv); zt = _l2norm(zt)

    # proxies concentrate on the first few (non-noisy) groups
    relevant_groups = [g for g in range(n_groups) if g not in noisy][:max(2, n_proxy)]
    proxy_emb = []
    for r in range(n_proxy):
        g = relevant_groups[r % len(relevant_groups)]
        Q = _l2norm(centers_t[g] + rng.normal(scale=0.12, size=(proxy_queries, d)))
        proxy_emb.append(Q.astype(np.float32))

    return {
        "z_v": zv, "z_t": zt, "proxy_emb": proxy_emb,
        "groups": groups, "noisy_groups": sorted(noisy),
        "relevant_groups": relevant_groups, "d": d, "n_groups": n_groups,
    }
