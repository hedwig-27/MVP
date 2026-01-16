import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


def _resolve_run_dir(path_arg, prefix, outputs_dir):
    if path_arg:
        p = Path(path_arg).expanduser()
        if p.is_file():
            return p.parent.resolve()
        if p.is_dir():
            return p.resolve()
    latest = outputs_dir / f"latest_{prefix}"
    if latest.exists():
        return latest.resolve()
    candidates = sorted(outputs_dir.glob(f"{prefix}_*"))
    return candidates[-1].resolve() if candidates else None


def _read_log(run_dir):
    if run_dir is None:
        return ""
    log_path = Path(run_dir) / "run.log"
    if not log_path.exists():
        return ""
    return log_path.read_text()


def _parse_metric(log_text, label):
    pattern = re.compile(rf"{re.escape(label)} \(([^)]+)\): ([0-9.]+)")
    results = {}
    for line in log_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        name = match.group(1)
        if name == "avg":
            continue
        results[name] = float(match.group(2))
    return results


def _parse_epoch_curve(log_text):
    pattern = re.compile(r"\[Epoch (\d+)\] Train Loss: ([0-9.]+), Val Loss: ([0-9.]+)")
    epochs = []
    train = []
    val = []
    for line in log_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        epochs.append(int(match.group(1)))
        train.append(float(match.group(2)))
        val.append(float(match.group(3)))
    return epochs, train, val


def _parse_val_by_epoch(log_text):
    val_re = re.compile(r"Val Loss \(([^)]+)\): ([0-9.]+)")
    epoch_re = re.compile(r"\[Epoch (\d+)\]")
    series = {}
    buffer = {}
    for line in log_text.splitlines():
        match = val_re.search(line)
        if match:
            buffer[match.group(1)] = float(match.group(2))
            continue
        match = epoch_re.search(line)
        if match:
            epoch = int(match.group(1))
            for name, val in buffer.items():
                entry = series.setdefault(name, {"epochs": [], "values": []})
                entry["epochs"].append(epoch)
                entry["values"].append(val)
            buffer = {}
    return series


def _parse_rollout(log_text):
    pattern = re.compile(r"MSE after (\d+) steps \(([^)]+)\): ([0-9.]+)")
    results = {}
    for line in log_text.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        step = int(match.group(1))
        name = match.group(2)
        val = float(match.group(3))
        results.setdefault(step, {})[name] = val
    return results


def _plot_compare(metric_a, metric_b, title, ylabel, out_path, label_a="primitive", label_b="fno"):
    names = sorted(set(metric_a) & set(metric_b))
    if not names:
        return False
    a_vals = [metric_a[n] for n in names]
    b_vals = [metric_b[n] for n in names]
    x = np.arange(len(names))
    width = 0.38

    plt.figure(figsize=(max(8, len(names) * 0.6), 4))
    plt.bar(x - width / 2, a_vals, width, label=label_a)
    plt.bar(x + width / 2, b_vals, width, label=label_b)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _plot_curve_compare(x_a, y_a, x_b, y_b, title, ylabel, out_path, label_a="primitive", label_b="fno"):
    if not x_a and not x_b:
        return False
    plt.figure(figsize=(6, 4))
    if x_a:
        plt.plot(x_a, y_a, marker="o", label=label_a)
    if x_b:
        plt.plot(x_b, y_b, marker="o", label=label_b)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _plot_train_val_curves(log_a, log_b, out_path, label_a="primitive", label_b="fno"):
    epochs_a, train_a, val_a = _parse_epoch_curve(log_a)
    epochs_b, train_b, val_b = _parse_epoch_curve(log_b)
    if not epochs_a and not epochs_b:
        return False
    plt.figure(figsize=(7, 6))
    plt.subplot(2, 1, 1)
    if epochs_a:
        plt.plot(epochs_a, train_a, marker="o", label=label_a)
    if epochs_b:
        plt.plot(epochs_b, train_b, marker="o", label=label_b)
    plt.ylabel("Train MSE")
    plt.title("Train Loss vs Epoch")
    plt.legend()
    plt.subplot(2, 1, 2)
    if epochs_a:
        plt.plot(epochs_a, val_a, marker="o", label=label_a)
    if epochs_b:
        plt.plot(epochs_b, val_b, marker="o", label=label_b)
    plt.xlabel("Epoch")
    plt.ylabel("Val MSE")
    plt.title("Val Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _plot_val_curves_by_dataset(series_a, series_b, out_dir, label_a="primitive", label_b="fno"):
    names = sorted(set(series_a) & set(series_b))
    if not names:
        return 0
    count = 0
    for name in names:
        epochs_a = series_a[name]["epochs"]
        vals_a = series_a[name]["values"]
        epochs_b = series_b[name]["epochs"]
        vals_b = series_b[name]["values"]
        if not epochs_a and not epochs_b:
            continue
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        out_path = out_dir / f"compare_val_curve_{safe}.png"
        _plot_curve_compare(
            epochs_a,
            vals_a,
            epochs_b,
            vals_b,
            f"Val Loss vs Epoch ({name})",
            "Val MSE",
            out_path,
            label_a=label_a,
            label_b=label_b,
        )
        count += 1
    return count


def _plot_single(metric, title, ylabel, out_path):
    if not metric:
        return False
    names = sorted(metric)
    vals = [metric[n] for n in names]
    x = np.arange(len(names))
    plt.figure(figsize=(max(8, len(names) * 0.6), 4))
    plt.bar(x, vals)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def _plot_usage(usage_path, out_path):
    if not usage_path.exists():
        return False
    data = json.loads(usage_path.read_text())
    if not data:
        return False
    num_primitives = len(data[0]["counts"])
    totals = [0] * num_primitives
    for item in data:
        for i, c in enumerate(item["counts"]):
            totals[i] += int(c)
    x = np.arange(1, num_primitives + 1)
    plt.figure(figsize=(6, 4))
    plt.bar(x, totals)
    plt.xticks(x, [f"P{i}" for i in x])
    plt.ylabel("Usage count")
    plt.title("Primitive usage (total)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots for MVP runs")
    parser.add_argument("--primitive", type=str, default=None, help="Primitive run dir or model path")
    parser.add_argument("--fno", type=str, default=None, help="FNO run dir or model path")
    parser.add_argument("--out", type=str, default=None, help="Output directory for plots")
    args = parser.parse_args()

    prim_dir = _resolve_run_dir(args.primitive, "primitive", OUTPUTS)
    fno_dir = _resolve_run_dir(args.fno, "fno", OUTPUTS)
    if prim_dir is None and fno_dir is None:
        print(f"No runs found under {OUTPUTS}. Run training first or pass --primitive/--fno.")
        return
    if args.out:
        out_dir = Path(args.out)
    else:
        out_dir = Path(prim_dir or fno_dir) / "plots"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prim_log = _read_log(prim_dir)
    fno_log = _read_log(fno_dir)

    prim_val = _parse_metric(prim_log, "Val Loss")
    fno_val = _parse_metric(fno_log, "Val Loss")
    _plot_compare(
        prim_val,
        fno_val,
        "Validation Loss (last epoch)",
        "MSE",
        out_dir / "compare_val_loss.png",
    )

    prim_test = _parse_metric(prim_log, "Test Loss")
    fno_test = _parse_metric(fno_log, "Test Loss")
    _plot_compare(
        prim_test,
        fno_test,
        "OOD Test Loss",
        "MSE",
        out_dir / "compare_ood_test_loss.png",
    )

    prim_mse = _parse_metric(prim_log, "One-step MSE")
    fno_mse = _parse_metric(fno_log, "One-step MSE")
    if prim_mse and fno_mse:
        _plot_compare(
            prim_mse,
            fno_mse,
            "One-step MSE (in-distribution)",
            "MSE",
            out_dir / "compare_onestep_mse.png",
        )
    elif prim_mse:
        _plot_single(
            prim_mse,
            "One-step MSE (primitive)",
            "MSE",
            out_dir / "primitive_onestep_mse.png",
        )

    if prim_dir:
        _plot_usage(Path(prim_dir) / "primitive_usage.json", out_dir / "primitive_usage.png")

    _plot_train_val_curves(
        prim_log,
        fno_log,
        out_dir / "compare_train_val_curve.png",
    )

    series_prim = _parse_val_by_epoch(prim_log)
    series_fno = _parse_val_by_epoch(fno_log)
    _plot_val_curves_by_dataset(series_prim, series_fno, out_dir)

    rollout_prim = _parse_rollout(prim_log)
    rollout_fno = _parse_rollout(fno_log)
    for step in sorted(set(rollout_prim) & set(rollout_fno)):
        _plot_compare(
            rollout_prim[step],
            rollout_fno[step],
            f"Rollout MSE at {step} steps",
            "MSE",
            out_dir / f"compare_rollout_{step}.png",
        )

    print(f"Plots saved to: {out_dir}")


if __name__ == "__main__":
    main()
