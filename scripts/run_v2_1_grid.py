"""Run a V2.1 alpha/lambda grid without colliding records.

Example:
  python scripts/run_v2_1_grid.py --config config.yaml \
    --alphas 0.6,0.75,0.9 --lambdas 0.5,0.7,0.9 \
    --budgets 0.05,0.1,0.2 --seeds 42 --epochs 1

Each grid point receives a V2.1 tag, so records.csv stores method labels such
as v2_1_a075_lv070. The underlying selector is always method v2_1.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _floats_csv(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _ints_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _tag(alpha: float, lambda_image: float) -> str:
    return f"a{int(round(alpha * 100)):03d}_lv{int(round(lambda_image * 100)):03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--alphas", default="0.6,0.75,0.9")
    ap.add_argument("--lambdas", default="0.5,0.7,0.9")
    ap.add_argument("--budgets", default=None, help="Optional CSV override, e.g. 0.05,0.1,0.2")
    ap.add_argument("--seeds", default=None, help="Optional CSV override, e.g. 42")
    ap.add_argument("--epochs", type=int, default=None, help="Optional train.num_epochs override")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    base_path = Path(args.config)
    cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    alphas = _floats_csv(args.alphas)
    lambdas = _floats_csv(args.lambdas)
    tmp_dir = Path("outputs") / "tmp_v2_1_grid_configs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for alpha in alphas:
        for lambda_image in lambdas:
            run_cfg = dict(cfg)
            run_cfg["coreset"] = dict(cfg["coreset"])
            run_cfg["coreset"]["methods"] = ["v2_1"]
            run_cfg["coreset"]["v2_1"] = dict(cfg["coreset"].get("v2_1", {}))
            run_cfg["coreset"]["v2_1"]["selection_alpha"] = alpha
            run_cfg["coreset"]["v2_1"]["lambda_image"] = lambda_image
            run_cfg["coreset"]["v2_1"]["tag"] = _tag(alpha, lambda_image)
            if args.budgets:
                run_cfg["coreset"]["budgets"] = _floats_csv(args.budgets)
            if args.seeds:
                run_cfg["train"] = dict(cfg["train"])
                run_cfg["train"]["seeds"] = _ints_csv(args.seeds)
            if args.epochs is not None:
                run_cfg["train"] = dict(run_cfg["train"])
                run_cfg["train"]["num_epochs"] = args.epochs

            out_path = tmp_dir / f"config_{run_cfg['coreset']['v2_1']['tag']}.yaml"
            out_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
            cmd = [sys.executable, "scripts/run_sweep.py", "--config", str(out_path)]
            print(" ".join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
