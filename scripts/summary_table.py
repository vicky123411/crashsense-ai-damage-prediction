"""Compact markdown table from results/benchmark_raw.json: damage MAE and level macro-F1 per split.

    python scripts/summary_table.py [results/benchmark_raw.json]
"""
import json
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/benchmark_raw.json"
with open(path) as f:
    res = json.load(f)

splits = list(res)
models = list(res[splits[0]])


def cell(runs, key, digits=3):
    vals = [r[key] for r in (runs or []) if key in r]
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.{digits}f}"
    return f"{np.mean(vals):.{digits}f} ± {np.std(vals):.{digits}f}"


for key, title, digits in [("damage_mae", "Damage MAE (mean absolute error of the 0-1 damage value, lower is better)", 3),
                           ("level_macro_f1", "Damage level macro-F1 (five levels, higher is better)", 3)]:
    print(f"**{title}**\n")
    print("| model | " + " | ".join(splits) + " |")
    print("|---|" + "---:|" * len(splits))
    for m in models:
        print(f"| {m} | " + " | ".join(cell(res[s].get(m), key, digits) for s in splits) + " |")
    print()
