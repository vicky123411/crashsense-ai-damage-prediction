"""Time the physics layer and a trained model on the same random scenarios (CPU only).

    python scripts/time_inference.py [--model models/gnn_demo.pt] [--n 1000] [--threads 1]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch  # noqa: E402

from crashdamage import dataset  # noqa: E402
from crashdamage.export import Predictor  # noqa: E402
from crashdamage.train import load_bundle, predict_arrays  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/gnn_demo.pt")
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = library default)")
args = ap.parse_args()
if args.threads > 0:
    torch.set_num_threads(args.threads)

rng = np.random.default_rng(123)
scenarios = [dataset.random_scenario(rng) for _ in range(args.n)]
physics, model = Predictor(), Predictor(args.model)
print(f"{args.n} random scenarios, {os.cpu_count()} CPU cores, torch threads: {torch.get_num_threads()}")


def ms_each(fn):
    t0 = time.perf_counter()
    for sc in scenarios:
        fn(sc)
    return (time.perf_counter() - t0) / len(scenarios) * 1e3


for sc in scenarios[:20]:                      # warm-up
    physics.predict(sc, "physics")
    model.predict(sc, "model")

print(f"physics layer, one request at a time:       {ms_each(lambda s: physics.predict(s, 'physics')):7.2f} ms")
print(f"trained model, one request at a time:       {ms_each(lambda s: model.predict(s, 'model')):7.2f} ms")

t0 = time.perf_counter()
inputs = [dataset.scenario_inputs(sc) for sc in scenarios]
t_feat = (time.perf_counter() - t0) / len(scenarios) * 1e3
nx = np.stack([i["node_x"] for i in inputs])
gx = np.stack([i["glob_x"] for i in inputs])
bundle = load_bundle(args.model)
if bundle.cfg.physics_features:
    nx = np.concatenate([nx, np.stack([i["phys_node"] for i in inputs])], axis=2)
    gx = np.concatenate([gx, np.stack([i["phys_glob"] for i in inputs])], axis=1)
t0 = time.perf_counter()
predict_arrays(bundle, nx, gx)
t_net = (time.perf_counter() - t0) / len(scenarios) * 1e3
print(f"model, whole batch at once: features {t_feat:.2f} ms + network {t_net:.3f} ms per scenario")
