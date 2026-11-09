"""Bar charts of the benchmark (results/benchmark_raw.json) -> results/benchmark.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

raw_path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_raw.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else "results/benchmark.png"
with open(raw_path) as f:
    res = json.load(f)

splits = list(res)
models = list(next(iter(res.values())))
metrics = [("damage_mae", "damage MAE (lower is better)"), ("level_macro_f1", "level macro-F1 (higher is better)")]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
width = 0.8 / len(models)
for ax, (key, title) in zip(axes, metrics):
    for k, m in enumerate(models):
        means, errs = [], []
        for s in splits:
            vals = [r[key] for r in res[s].get(m, []) if key in r]
            means.append(np.mean(vals) if vals else np.nan)
            errs.append(np.std(vals) if len(vals) > 1 else 0.0)
        ax.bar(np.arange(len(splits)) + k * width, means, width, yerr=errs, label=m, capsize=2)
    ax.set_xticks(np.arange(len(splits)) + 0.4 - width / 2)
    ax.set_xticklabels(splits)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=len(models), fontsize=8, frameon=False)
fig.suptitle("Agreement with the physics layer on synthetic data (mean ± std over seeds)", fontsize=10)
fig.tight_layout(rect=(0, 0.08, 1, 0.95))
fig.savefig(out_path, dpi=150)
print("wrote", out_path)
