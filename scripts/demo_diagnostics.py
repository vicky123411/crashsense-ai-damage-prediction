"""Where does the demo model make its errors?

    python scripts/demo_diagnostics.py                     # writes results/demo_diagnostics.json

Re-scores models/gnn_demo.pt on the test rows of the random split (seed 0) of data/synthetic.npz, prints the headline
numbers (they should match the end of results/demo_model_metrics.txt) and breaks the damage error down by part kind,
speed, obstacle preset and face, plus a level confusion matrix.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from crashdamage import dataset  # noqa: E402
from crashdamage.metrics import evaluate_predictions, level_from_damage  # noqa: E402
from crashdamage.obstacles import PRESET_NAMES  # noqa: E402
from crashdamage.train import load_bundle, predict  # noqa: E402
from crashdamage.vehicle import FACES, KINDS, PART_NAMES, PART_TEMPLATE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synthetic.npz"))
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "gnn_demo.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "demo_diagnostics.json"))
    args = ap.parse_args()

    data = dataset.load(args.data)
    split = dataset.make_split(data, "random", seed=0)
    test = split[2]
    bundle = load_bundle(args.model)
    pred = predict(bundle, data, test)
    metrics = evaluate_predictions(data, test, pred)

    d_true = data["damage"][test]
    d_pred = np.clip(pred["damage"], 0, 1)
    err = np.abs(d_pred - d_true)
    out = {"n_train": int(len(split[0])), "n_val": int(len(split[1])), "n_test": int(len(test)),
           "headline": {k: metrics[k] for k in ("damage_mae", "damage_r2", "level_acc", "level_within_one",
                                                "level_macro_f1", "detach_f1", "crush_depth_m_mae", "intrusion_m_mae",
                                                "obstacle_damage_mae", "damage_mae_loaded", "damage_mae_unloaded")}}

    kinds = np.array([p.kind for p in PART_TEMPLATE])
    out["mae_by_kind"] = {k: float(err[:, kinds == k].mean()) for k in KINDS}
    out["n_parts_by_kind"] = {k: int((kinds == k).sum()) for k in KINDS}
    out["mae_by_part"] = {n: float(err[:, i].mean()) for i, n in enumerate(PART_NAMES)}

    speed = data["speed"][test]
    bins = [4, 12, 20, 28, 36.0001]
    out["mae_by_speed_ms"] = {f"{bins[i]:.0f}-{min(bins[i + 1], 36):.0f}": float(err[(speed >= bins[i]) & (speed < bins[i + 1])].mean())
                              for i in range(4)}
    out["n_by_speed_ms"] = {f"{bins[i]:.0f}-{min(bins[i + 1], 36):.0f}": int(((speed >= bins[i]) & (speed < bins[i + 1])).sum())
                            for i in range(4)}

    preset = data["preset"][test]
    out["mae_by_obstacle"] = {n: float(err[preset == k].mean()) for k, n in enumerate(PRESET_NAMES) if (preset == k).any()}
    out["n_by_obstacle"] = {n: int((preset == k).sum()) for k, n in enumerate(PRESET_NAMES)}
    reg_err = np.abs(pred["reg"][:, 3] - data["glob_reg"][test][:, 3])
    out["obstacle_damage_mae_by_obstacle"] = {n: float(reg_err[preset == k].mean()) for k, n in enumerate(PRESET_NAMES)
                                              if (preset == k).any()}
    face = data["face"][test]
    out["mae_by_face"] = {f: float(err[face == k].mean()) for k, f in enumerate(FACES) if (face == k).any()}
    out["n_by_face"] = {f: int((face == k).sum()) for k, f in enumerate(FACES)}

    lv_true = data["level"][test].ravel()
    lv_pred = level_from_damage(d_pred).ravel()
    conf = np.zeros((5, 5), dtype=int)
    for a, b in zip(lv_true, lv_pred):
        conf[a, b] += 1
    out["level_confusion_rows_true"] = conf.tolist()
    out["level_share_true"] = (conf.sum(axis=1) / conf.sum()).tolist()

    flat = err.ravel()
    out["abs_error_share_below"] = {str(t): float((flat < t).mean()) for t in (0.01, 0.05, 0.1, 0.2)}
    out["abs_error_quantiles"] = {str(q): float(np.quantile(flat, q)) for q in (0.5, 0.9, 0.95, 0.99)}
    scen = err.mean(axis=1)
    worst = np.argsort(-scen)[:8]
    out["worst_scenarios"] = [dict(row=int(test[i]), mae=float(scen[i]), obstacle=PRESET_NAMES[int(preset[i])],
                                   face=FACES[int(face[i])], speed_ms=float(speed[i])) for i in worst]
    out["scenario_mae_quantiles"] = {str(q): float(np.quantile(scen, q)) for q in (0.5, 0.9, 0.99)}

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out["headline"].items()}, indent=1))
    print("by kind  ", {k: round(v, 4) for k, v in out["mae_by_kind"].items()})
    print("by speed ", {k: round(v, 4) for k, v in out["mae_by_speed_ms"].items()})
    print("by obst. ", {k: round(v, 4) for k, v in out["mae_by_obstacle"].items()})
    print("by face  ", {k: round(v, 4) for k, v in out["mae_by_face"].items()})
    print("errors   ", out["abs_error_share_below"], out["abs_error_quantiles"])
    print("worst    ", out["worst_scenarios"][:4])
    print("wrote", args.out)


if __name__ == "__main__":
    main()
