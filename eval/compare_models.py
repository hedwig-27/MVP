import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import setup_logger


def _load_metrics(run_dir, split):
    path = Path(run_dir) / "plots" / "eval" / f"eval_metrics_{split}.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def _plot_compare(names, prim_vals, fno_vals, title, out_path, ylabel):
    if not names:
        return
    x = list(range(len(names)))
    width = 0.35
    plt.figure(figsize=(max(7, len(names) * 0.6), 4))
    plt.bar([i - width / 2 for i in x], prim_vals, width=width, label="Primitive")
    plt.bar([i + width / 2 for i in x], fno_vals, width=width, label="FNO")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _avg(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description="Compare Primitive vs FNO eval metrics")
    parser.add_argument("--primitive_run", type=str, default="outputs/latest_primitive")
    parser.add_argument("--fno_run", type=str, default="outputs/latest_fno")
    args = parser.parse_args()

    primitive_run = Path(args.primitive_run)
    fno_run = Path(args.fno_run)
    if not primitive_run.exists() or not fno_run.exists():
        raise FileNotFoundError("primitive_run or fno_run does not exist.")

    out_dir = primitive_run / "plots" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger, log_path = setup_logger("compare", primitive_run)
    logger.info("Log file: %s", log_path)

    splits = []
    for split in ("id", "ood_param", "ood_equation"):
        if (primitive_run / "plots" / "eval" / f"eval_metrics_{split}.json").exists() and (
            fno_run / "plots" / "eval" / f"eval_metrics_{split}.json"
        ).exists():
            splits.append(split)

    summary = {"primitive_run": str(primitive_run), "fno_run": str(fno_run), "splits": {}}
    for split in splits:
        prim = _load_metrics(primitive_run, split)
        fno = _load_metrics(fno_run, split)
        if prim is None or fno is None:
            continue

        prim_one = prim.get("one_step", {})
        fno_one = fno.get("one_step", {})
        names = [n for n in prim_one.keys() if n in fno_one]
        prim_one_vals = [prim_one[n] for n in names]
        fno_one_vals = [fno_one[n] for n in names]

        summary["splits"][split] = {
            "one_step": {"primitive": prim_one, "fno": fno_one, "avg": {"primitive": _avg(prim_one_vals), "fno": _avg(fno_one_vals)}},
        }

        _plot_compare(
            names,
            prim_one_vals,
            fno_one_vals,
            f"One-step NRMSE compare ({split})",
            out_dir / f"compare_onestep_{split}.png",
            "NRMSE",
        )
        logger.info("[%s] One-step avg: Primitive=%.6f FNO=%.6f", split.upper(), _avg(prim_one_vals), _avg(fno_one_vals))

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Comparison complete. Plots saved to %s", out_dir)


if __name__ == "__main__":
    main()
