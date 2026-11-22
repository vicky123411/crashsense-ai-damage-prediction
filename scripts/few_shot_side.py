"""How many crashes on an unseen side does the model need?

The 'face' split never shows a rear impact in training. Here k rear crashes (k = 0, 25, 50, 100, 200)
are moved from the test set into the training set; the remaining rear crashes (the same rows for
every k) are the test set. Because labels come from the physics layer, adding crashes on a missing
side costs seconds, so this measures what it takes to close the gap.

    python scripts/few_shot_side.py --seeds 0,1,2 --models gnn,node_mlp,gnn_imp --out results/few_shot_side.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crashdamage import dataset  # noqa: E402
from crashdamage.evaluate import variant_config  # noqa: E402
from crashdamage.metrics import evaluate_predictions  # noqa: E402
from crashdamage.train import predict, train  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/synthetic.npz")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--models", default="gnn,node_mlp")
ap.add_argument("--ks", default="0,25,50,100,200")
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--out", default="results/few_shot_side.json")
args = ap.parse_args()

data = dataset.load(args.data)
ks = [int(k) for k in args.ks.split(",")]
results = {}
if os.path.exists(args.out):
    with open(args.out) as f:
        results = json.load(f)
t0 = time.time()
for seed in [int(s) for s in args.seeds.split(",")]:
    tr, va, te = dataset.make_split(data, "face", seed=0)
    te = np.random.default_rng(100 + seed).permutation(te)
    test = te[max(ks):]                                   # identical test rows for every k
    for k in ks:
        split = (np.concatenate([tr, te[:k]]), va, test)
        for m in args.models.split(","):
            key = f"{m}|{k}|{seed}"
            if key in results:
                continue
            cfg = variant_config(m, args.epochs)
            cfg.seed = seed
            bundle, _ = train(data, split, cfg)
            results[key] = evaluate_predictions(data, test, predict(bundle, data, test))
            print(f"[{time.time() - t0:5.0f}s] seed={seed} k={k:3d} {m:12s} damage MAE "
                  f"{results[key]['damage_mae'] * 100:.2f}", flush=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
