#!/usr/bin/env python
"""
scripts/embed_qproxy_families.py
================================
Encode the LLM-generated proxy query FAMILIES (raw text JSON) into CLIP text
embeddings (.npy), so they can be fed to scripts/run_ablation.py via --proxies.

This is the missing link: `qproxy_v3.py` produces 6 families of *text* queries
(JSON arrays of strings); the V3 selection pipeline needs each family as a
(m_r, d) array of *embeddings*. This script does that conversion.

------------------------------------------------------------------------------
⚠️  CRITICAL — encode proxies with the SAME CLIP model that produced z_v.npy /
    z_t.npy (the pool embeddings).
    MMD compares proxy embeddings against the pool's *text* embeddings in the
    SAME space. If the pool was encoded with open_clip ViT-B-32 (the V0–V2
    pipeline) but proxies with a different model, the spaces are misaligned.
    Both happen to be 512-d, so a mismatch will NOT raise an error — it will
    silently produce garbage relevance. Match the encoder exactly.
------------------------------------------------------------------------------

USAGE
-----
1) Dry run (NO torch/GPU needed) — just parse the JSON and report counts:
       python scripts/embed_qproxy_families.py \
           --proxy-dir outputs/qproxy_v3_hybrid6_20k --out-dir cache --dry-run

2) Encode (open_clip, matches the V0–V2 pool encoder ViT-B-32 / openai):
       python scripts/embed_qproxy_families.py \
           --proxy-dir outputs/qproxy_v3_hybrid6_20k --out-dir cache \
           --backend open_clip --model ViT-B-32 --pretrained openai

   (If your pool was encoded with openai/CLIP ViT-B/16 instead:)
       ... --backend openai_clip --model ViT-B/16

It prints the exact --proxies line to paste into run_ablation.py.
"""
from __future__ import annotations

import argparse
import json
import os
import glob

import numpy as np


# --------------------------------------------------------------------------- #
#  Family discovery + robust text extraction                                    #
# --------------------------------------------------------------------------- #
def discover_families(proxy_dir: str) -> dict[str, str]:
    """Return {family_name: json_path}. Prefers manifest.json; else globs *.json."""
    manifest = os.path.join(proxy_dir, "manifest.json")
    if os.path.exists(manifest):
        m = json.load(open(manifest, encoding="utf-8"))
        ff = m.get("family_files", {})
        if ff:
            return {name: os.path.join(proxy_dir, fn) for name, fn in ff.items()}
    # fallback: every top-level *.json except manifest, in non-checkpoint dir
    out = {}
    for p in sorted(glob.glob(os.path.join(proxy_dir, "*.json"))):
        base = os.path.basename(p)
        if base == "manifest.json":
            continue
        out[os.path.splitext(base)[0]] = p
    return out


def extract_texts(json_path: str) -> list[str]:
    """Extract query strings, robust to a few schemas.

    Confirmed schema for this repo: a plain JSON array of strings.
    Also handles {"queries":[...]} / list-of-dicts with a query/text key.
    """
    obj = json.load(open(json_path, encoding="utf-8"))
    items = None
    if isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        for k in ("queries", "texts", "items", "data"):
            if isinstance(obj.get(k), list):
                items = obj[k]
                break
    if items is None:
        raise ValueError(f"Unrecognized JSON schema in {json_path}")

    texts = []
    for x in items:
        if isinstance(x, str):
            s = x
        elif isinstance(x, dict):
            s = x.get("query") or x.get("text") or x.get("caption") or ""
        else:
            s = str(x)
        s = s.strip()
        if s:
            texts.append(s)
    # de-dup within family, keep order
    seen, uniq = set(), []
    for s in texts:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


# --------------------------------------------------------------------------- #
#  Encoders (load model ONCE, encode all families)                              #
# --------------------------------------------------------------------------- #
def _l2norm(x: np.ndarray) -> np.ndarray:
    return (x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)).astype(np.float32)


def make_encoder(backend: str, model: str, pretrained: str, device: str | None,
                 batch: int):
    """Return a function texts:list[str] -> (n, d) L2-normalized float32 array."""
    if backend == "open_clip":
        import torch, open_clip
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net, _, _ = open_clip.create_model_and_transforms(model, pretrained=pretrained)
        tok = open_clip.get_tokenizer(model)
        net = net.to(dev).eval()

        def enc(texts):
            outs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch):
                    t = tok(texts[i:i + batch]).to(dev)
                    outs.append(net.encode_text(t).float().cpu().numpy())
            return _l2norm(np.concatenate(outs, axis=0))
        return enc

    if backend == "openai_clip":
        import torch, clip
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net, _ = clip.load(model, device=dev)
        net.eval()

        def enc(texts):
            outs = []
            with torch.no_grad():
                for i in range(0, len(texts), batch):
                    t = clip.tokenize(texts[i:i + batch], truncate=True).to(dev)
                    outs.append(net.encode_text(t).float().cpu().numpy())
            return _l2norm(np.concatenate(outs, axis=0))
        return enc

    raise ValueError(f"unknown backend {backend}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", required=True,
                    help="dir with family JSONs + manifest.json (e.g. outputs/qproxy_v3_hybrid6_20k)")
    ap.add_argument("--out-dir", default="cache", help="where to write <family>.npy")
    ap.add_argument("--backend", choices=["open_clip", "openai_clip"], default="open_clip")
    ap.add_argument("--model", default="ViT-B-32",
                    help="open_clip: 'ViT-B-32'; openai_clip: 'ViT-B/16'")
    ap.add_argument("--pretrained", default="openai", help="open_clip pretrained tag")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--dry-run", action="store_true",
                    help="parse JSON + report counts, do NOT load CLIP or encode")
    args = ap.parse_args()

    families = discover_families(args.proxy_dir)
    if not families:
        raise SystemExit(f"No family JSONs found in {args.proxy_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[info] found {len(families)} families in {args.proxy_dir}")
    parsed = {}
    for name, path in families.items():
        texts = extract_texts(path)
        parsed[name] = texts
        print(f"  - {name:28s} {len(texts):6d} queries  ({os.path.basename(path)})")

    if args.dry_run:
        print("[dry-run] parsing OK; no embeddings written. "
              "Re-run without --dry-run to encode.")
        return

    print(f"[info] loading encoder: backend={args.backend} model={args.model} "
          f"pretrained={args.pretrained}")
    print("[warn] proxies MUST use the SAME CLIP model that encoded z_v.npy/z_t.npy "
          "(same dim is NOT enough — must be same weights).")
    enc = make_encoder(args.backend, args.model, args.pretrained, args.device, args.batch)

    npy_paths = []
    dims = set()
    for name, texts in parsed.items():
        emb = enc(texts)                       # (m_r, d) normalized
        out = os.path.join(args.out_dir, f"{name}.npy")
        np.save(out, emb)
        npy_paths.append(out)
        dims.add(emb.shape[1])
        print(f"[ok] {name:28s} -> {out}  shape={emb.shape}")

    if len(dims) > 1:
        print(f"[ERROR] families have inconsistent embedding dims {dims} — check encoder.")
    print("\n=== paste this into run_ablation.py ===")
    print("  --proxies " + " ".join(npy_paths))
    print("(and make sure --zv/--zt were encoded with the SAME model)")


if __name__ == "__main__":
    main()
