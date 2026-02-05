#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml


def deep_merge(base, override):
    if not isinstance(base, dict) or not isinstance(override, dict):
        return override
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def list_runs(outputs_dir):
    if not outputs_dir.exists():
        return {}
    return {
        p.name: p
        for p in outputs_dir.iterdir()
        if p.is_dir() and p.name.startswith("primitive_")
    }


def run_and_log(cmd, cwd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        env = dict(os.environ)
        env.setdefault("TQDM_FORCE_TTY", "1")
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            sys.stdout.write(chunk)
            sys.stdout.flush()
            f.write(chunk)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"Command failed with code {ret}: {' '.join(cmd)}")


def load_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compute_avg(metrics):
    if not metrics:
        return None
    vals = [float(v) for v in metrics.values()]
    return sum(vals) / max(len(vals), 1)


def main():
    parser = argparse.ArgumentParser(description="Run experiment suite and compare results.")
    parser.add_argument(
        "--suite",
        default="configs/experiment_suite.yaml",
        help="Path to experiment suite yaml.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    suite_path = (repo_root / args.suite).resolve()
    suite = yaml.safe_load(suite_path.read_text(encoding="utf-8"))

    base_config_path = (repo_root / suite["base_config"]).resolve()
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))

    out_root = repo_root / suite.get("output_dir", "outputs/experiment_suites")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    suite_dir = out_root / f"suite_{stamp}"
    cfg_dir = suite_dir / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)

    experiments = suite.get("experiments", [])
    if not experiments:
        raise RuntimeError("No experiments defined in suite.")

    outputs_dir = repo_root / "outputs"
    results = []

    for exp in experiments:
        name = exp["name"]
        overrides = exp.get("overrides", {})
        exp_cfg = deep_merge(base_config, overrides)

        cfg_path = cfg_dir / f"{name}.yaml"
        cfg_path.write_text(yaml.safe_dump(exp_cfg, sort_keys=False), encoding="utf-8")

        before = list_runs(outputs_dir)

        print(f"\n=== Running {name} ===")
        run_and_log(
            [sys.executable, "train/train_primitives.py", "--config", str(cfg_path)],
            cwd=repo_root,
            log_path=suite_dir / f"{name}_train.log",
        )

        after = list_runs(outputs_dir)
        new_dirs = [after[k] for k in after.keys() - before.keys()]
        if len(new_dirs) == 1:
            run_dir = new_dirs[0]
        else:
            run_dir = max(after.values(), key=lambda p: p.stat().st_mtime)

        model_path = run_dir / "primitive_model.pt"
        if not model_path.exists():
            raise RuntimeError(f"Model not found: {model_path}")

        run_and_log(
            [sys.executable, "eval/eval_suite.py", "--config", str(cfg_path), "--model", str(model_path)],
            cwd=repo_root,
            log_path=suite_dir / f"{name}_eval.log",
        )

        train_metrics = load_json(run_dir / "primitive_metrics.json")
        id_metrics = load_json(run_dir / "plots" / "eval" / "eval_metrics_id.json")
        ood_param = load_json(run_dir / "plots" / "eval" / "eval_metrics_ood_param.json")
        ood_eq = load_json(run_dir / "plots" / "eval" / "eval_metrics_ood_equation.json")

        id_avg = id_metrics.get("one_step_avg") or compute_avg(id_metrics.get("one_step", {}))
        ood_param_avg = ood_param.get("one_step_avg") or compute_avg(ood_param.get("one_step", {}))
        ood_eq_avg = ood_eq.get("one_step_avg") or compute_avg(ood_eq.get("one_step", {}))

        overall_parts = [v for v in [id_avg, ood_param_avg, ood_eq_avg] if v is not None]
        overall = sum(overall_parts) / max(len(overall_parts), 1) if overall_parts else None

        results.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "train_loss": train_metrics.get("train_loss"),
                "val_loss": train_metrics.get("val_loss"),
                "id_avg": id_avg,
                "ood_param_avg": ood_param_avg,
                "ood_eq_avg": ood_eq_avg,
                "overall": overall,
            }
        )

    # Rank by overall (lower is better)
    ranked = sorted([r for r in results if r["overall"] is not None], key=lambda r: r["overall"])
    best = ranked[0] if ranked else None

    summary = {
        "suite": str(suite_path),
        "runs": results,
        "best": best,
    }
    (suite_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Write summary table
    lines = [
        "# Experiment Summary",
        "",
        "| name | train | val | id_avg | ood_param_avg | ood_eq_avg | overall | run_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            "| {name} | {train_loss:.6f} | {val_loss:.6f} | {id_avg:.6f} | {ood_param_avg:.6f} | {ood_eq_avg:.6f} | {overall:.6f} | {run_dir} |".format(
                name=r["name"],
                train_loss=float(r["train_loss"] or 0.0),
                val_loss=float(r["val_loss"] or 0.0),
                id_avg=float(r["id_avg"] or 0.0),
                ood_param_avg=float(r["ood_param_avg"] or 0.0),
                ood_eq_avg=float(r["ood_eq_avg"] or 0.0),
                overall=float(r["overall"] or 0.0),
                run_dir=r["run_dir"],
            )
        )
    if best:
        lines += ["", f"Best: {best['name']} (overall={best['overall']:.6f})"]
    (suite_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n=== Summary ===")
    for r in results:
        print(f"{r['name']}: overall={r['overall']:.6f} run_dir={r['run_dir']}")
    if best:
        print(f"Best: {best['name']} (overall={best['overall']:.6f})")


if __name__ == "__main__":
    main()
