"""Compare the models on the standard split types and write a results table."""
import json
import time

import numpy as np

from . import dataset
from .baselines import GradientBoostingBaseline, NaiveEnergyBaseline
from .metrics import HEADLINE, evaluate_predictions
from .train import TrainConfig, predict, train

SPLITS = ["random", "speed", "obstacle", "face"]

# name -> description shown in the tables
VARIANT_INFO = {
    "naive_energy": "energy shared by capacity, no learning",
    "gbm": "gradient boosting (scikit-learn)",
    "mlp": "flat MLP, no graph",
    "node_mlp": "per-part MLP, no message passing",
    "gnn": "graph network",
    "gnn_phys": "graph network + energy/capacity inputs",
    "gnn_phys_energy": "graph network + energy inputs + energy-consistency loss",
}


def variant_config(name, epochs):
    table = {
        "mlp": dict(kind="mlp"),
        "node_mlp": dict(kind="node_mlp"),
        "gnn": dict(kind="gnn"),
        "gnn_phys": dict(kind="gnn", physics_features=True),
        "gnn_phys_energy": dict(kind="gnn", physics_features=True, energy_loss=1.0),
    }
    return TrainConfig(epochs=epochs, **table[name])


def run_one(data, split, variant, seed, epochs):
    """Train / fit one variant on one split and score it on the test rows."""
    test = split[2]
    if variant == "naive_energy":
        pred = NaiveEnergyBaseline().predict(data, test)
    elif variant == "gbm":
        pred = GradientBoostingBaseline(seed=seed).fit(data, split[0]).predict(data, test)
    else:
        cfg = variant_config(variant, epochs)
        cfg.seed = seed
        bundle, _ = train(data, split, cfg)
        pred = predict(bundle, data, test)
    return evaluate_predictions(data, test, pred)


def run_benchmark(data, variants, splits=SPLITS, seeds=(0, 1, 2), epochs=40, out_path=None, log=print):
    """results[split][variant] = list of metric dicts, one per seed."""
    results = {}
    t0 = time.time()
    for sk in splits:
        results[sk] = {}
        for v in variants:
            runs = []
            seed_list = (0,) if v in ("naive_energy", "gbm") else seeds   # these two do not depend on the seed much
            for seed in seed_list:
                split = dataset.make_split(data, sk, seed=seed)
                runs.append(run_one(data, split, v, seed, epochs))
                log(f"[{time.time() - t0:6.0f}s] split={sk:8s} model={v:16s} seed={seed} "
                    f"damage_mae={runs[-1]['damage_mae']:.4f}")
            results[sk][v] = runs
            if out_path:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=1)
    return results


def summarize(runs, key):
    vals = [r[key] for r in runs if key in r]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def markdown_table(results, split, keys=HEADLINE):
    names = {"damage_mae": "damage MAE", "damage_r2": "damage R²", "level_macro_f1": "level macro-F1",
             "detach_f1": "detach F1", "crush_depth_m_mae": "crush depth MAE (m)",
             "obstacle_damage_mae": "obstacle damage MAE", "through_f1": "through-car F1"}
    head = "| model | " + " | ".join(names.get(k, k) for k in keys) + " |"
    rule = "|---|" + "---:|" * len(keys)
    rows = [head, rule]
    for v, runs in results[split].items():
        cells = []
        for k in keys:
            s = summarize(runs, k)
            if s is None:
                cells.append("–")
            elif len(runs) > 1:
                cells.append(f"{s[0]:.3f} ± {s[1]:.3f}")
            else:
                cells.append(f"{s[0]:.3f}")
        rows.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(rows)
