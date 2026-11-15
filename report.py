import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from calibration import *
from utils import read_jsonl, save_json


def load_preds(name, split):
    rows = read_jsonl(PREDS / (name + "_" + split + ".jsonl"))
    rows.sort(key=lambda r: r["id"])
    return rows


def arrays(rows):
    d = np.array([r["logit_diff"] for r in rows])
    y = np.array([r["label"] for r in rows])
    return d, y


def score_model(name):
    d_val, y_val = arrays(load_preds(name, "val"))
    T = fit_temperature(d_val, y_val)
    out = {"temperature": round(T, 4)}
    for split in ["val", "test", "test_unseen"]:
        rows = load_preds(name, split)
        d, y = arrays(rows)
        out[split] = {"raw": all_metrics(d, y, bins=BINS), "temp_scaled": all_metrics(d, y, T, bins=BINS)}
        out[split]["other_mass"] = round(float(np.mean([r["other_mass"] for r in rows])), 4)
    return out, T


def compare(a, b, T_a, T_b, split):
    rows_a = load_preds(a, split)
    rows_b = load_preds(b, split)
    assert [r["id"] for r in rows_a] == [r["id"] for r in rows_b]
    d_a, y = arrays(rows_a)
    d_b, _ = arrays(rows_b)
    groups = [r["image"] for r in rows_a]
    return {"raw": bootstrap_diff(groups, y, d_a, d_b, bins=BINS),
            "temp_scaled": bootstrap_diff(groups, y, d_a, d_b, T_a, T_b, bins=BINS)}


def by_relation(name, split, min_rows=40):
    groups = {}
    for r in load_preds(name, split):
        groups.setdefault(r["relation"], []).append(r)
    out = {}
    for rel in sorted(groups):
        if len(groups[rel]) < min_rows:
            continue
        d, y = arrays(groups[rel])
        out[rel] = {"n": len(y), "accuracy": round(accuracy(p_yes(d), y), 4)}
    return out


def reliability_plot(names, temps, split, path):
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4.2))
    if len(names) == 1:
        axes = [axes]
    for k in range(len(names)):
        ax = axes[k]
        d, y = arrays(load_preds(names[k], split))
        ax.plot([0.5, 1], [0.5, 1], color="gray", linestyle="--", label="perfect")
        for T, label in [(1.0, "raw"), (temps[k], "temp scaled")]:
            xs = []
            ys = []
            for b in bin_stats(p_yes(d, T), y, BINS):
                if b["n"] >= 10:
                    xs.append(b["conf"])
                    ys.append(b["acc"])
            ax.plot(xs, ys, marker="o", label=label)
        ax.set_title(names[k] + " (" + split + ")")
        ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print("wrote", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="base")
    ap.add_argument("--main", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    args = ap.parse_args()

    report = {"models": {}, "base_vs_main": {}, "by_relation": {}}
    temps = {}
    for name in [args.base, args.main] + args.others:
        report["models"][name], temps[name] = score_model(name)
        print(name, "T", report["models"][name]["temperature"])
        for split in ["test", "test_unseen"]:
            print("  ", split, "raw", report["models"][name][split]["raw"])
            print("  ", split, "scaled", report["models"][name][split]["temp_scaled"])

    for split in ["test", "test_unseen"]:
        report["base_vs_main"][split] = compare(args.base, args.main, temps[args.base], temps[args.main], split)
        report["by_relation"][split] = {"base": by_relation(args.base, split), "main": by_relation(args.main, split)}
        print(split, "main minus base", report["base_vs_main"][split])

    save_json(report, RESULTS / "report.json")
    reliability_plot([args.base, args.main], [temps[args.base], temps[args.main]], "test", RESULTS / "reliability.png")