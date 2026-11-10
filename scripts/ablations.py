"""Extra experiments on the graph network: how deep it needs to be, whether the car's real part graph matters,
and how much labelled data it needs.

    python scripts/ablations.py                    # run everything (about an hour on 2 CPU cores), resumes if stopped
    python scripts/ablations.py --quick            # 2 epochs of one job, to test the script
    python scripts/ablations.py --workers 1

Everything uses the data in data/synthetic.npz (python -m crashdamage generate), the same training settings as the
benchmark (40 epochs at most, early stopping, 3 seeds) and the same four metrics code. Results are appended to
results/ablations.json, one record per run.

Experiments
  depth   graph network with 1, 2 or 4 message-passing rounds (0 = node_mlp and 3 = gnn are in the benchmark);
          5 and 6 rounds on the random split only
  graph   graph network (3 rounds) with the car's part graph replaced by
            no_edges : no connections at all (same layers and parameters, parts never see each other)
            complete : every part is joined to every other part with the same weight
            shuffled : the real edge weights, but between parts chosen at random (new draw for every seed)
  size    random split, only the first n training scenarios are used (the validation set shrinks with it)
  repro   the benchmark's gnn on the random split with seed 0, to check that the numbers can be reproduced
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

from crashdamage import dataset, models, vehicle  # noqa: E402
from crashdamage.baselines import GradientBoostingBaseline  # noqa: E402
from crashdamage.metrics import evaluate_predictions  # noqa: E402
from crashdamage.train import TrainConfig, predict, train  # noqa: E402

DATA = None            # filled in by main(); worker processes inherit it
EPOCHS = 40
SEEDS = (0, 1, 2)


def adjacency_for(name, seed):
    a = vehicle.adjacency_matrix()
    n = a.shape[0]
    if name == "physical":
        return a
    if name == "no_edges":
        return np.zeros_like(a)
    if name == "complete":
        return (np.ones((n, n)) - np.eye(n)).astype(np.float32)
    if name == "shuffled":
        perm = np.random.default_rng(1000 + seed).permutation(n)
        return a[np.ix_(perm, perm)]
    raise ValueError(name)


def make_jobs():
    """In the order they are run, most useful first (so a stopped run still leaves a usable set)."""
    def graph_jobs(split):
        return [dict(kind="graph", split=split, variant="gnn", layers=3, graph=graph, seed=seed)
                for graph in ("no_edges", "complete", "shuffled") for seed in SEEDS]

    def depth_jobs(split, depths=(1, 2, 4)):
        return [dict(kind="depth", split=split, variant="gnn", layers=layers, graph="physical", seed=seed)
                for layers in depths for seed in SEEDS]

    size_jobs = [dict(kind="size", split="random", variant=variant, layers=3 if variant == "gnn" else 0,
                      graph="physical", seed=seed, n_train=n_train)
                 for n_train in (250, 500, 1000, 2000) for variant in ("gnn", "node_mlp", "gbm") for seed in SEEDS]
    repro = [dict(kind="repro", split="random", variant="gnn", layers=3, graph="physical", seed=0)]
    return (repro + graph_jobs("random") + depth_jobs("random") + size_jobs + graph_jobs("speed")
            + depth_jobs("speed") + graph_jobs("obstacle") + depth_jobs("random", depths=(5, 6)))


def job_key(j):
    return (j["kind"], j["split"], j["variant"], j["layers"], j["graph"], j["seed"], j.get("n_train", 0))


def init_worker():
    import torch
    torch.set_num_threads(1)


def run_job(job):
    t0 = time.time()
    epochs = job.get("epochs", EPOCHS)
    split = dataset.make_split(DATA, job["split"], seed=job["seed"])
    if "n_train" in job:
        n = job["n_train"]
        n_val = max(50, int(round(n * 15 / 70)))
        split = (split[0][:n], split[1][:n_val], split[2])
    test = split[2]
    original = models.adjacency_matrix
    models.adjacency_matrix = lambda: adjacency_for(job["graph"], job["seed"])
    try:
        if job["variant"] == "gbm":
            pred = GradientBoostingBaseline(seed=job["seed"]).fit(DATA, split[0]).predict(DATA, test)
            ran = None
        else:
            kind = "gnn" if job["variant"] == "gnn" else "node_mlp"
            cfg = TrainConfig(kind=kind, epochs=epochs, seed=job["seed"])
            if kind == "gnn":
                cfg.layers = job["layers"]
            bundle, hist = train(DATA, split, cfg)
            pred = predict(bundle, DATA, test)
            ran = len(hist)
    finally:
        models.adjacency_matrix = original
    rec = dict(job)
    rec.update(evaluate_predictions(DATA, test, pred))
    rec["epochs_run"] = ran
    rec["n_test"] = int(len(test))
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


def main():
    global DATA, EPOCHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synthetic.npz"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ablations.json"))
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true", help="one tiny job (2 epochs), results are not saved")
    args = ap.parse_args()

    DATA = dataset.load(args.data)
    if args.quick:
        EPOCHS = 2
        job = dict(kind="graph", split="random", variant="gnn", layers=3, graph="shuffled", seed=0)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in run_job(job).items()}, indent=1))
        return

    done = []
    if os.path.exists(args.out):
        with open(args.out) as f:
            done = json.load(f)
    have = {job_key(r) for r in done}
    todo = [j for j in make_jobs() if job_key(j) not in have]
    print(f"{len(done)} runs already done, {len(todo)} to do", flush=True)
    t0 = time.time()
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=init_worker) as pool:
        for rec in pool.imap_unordered(run_job, todo):
            done.append(rec)
            with open(args.out, "w") as f:
                json.dump(done, f, indent=1)
            print(f"[{time.time() - t0:6.0f}s] {rec['kind']:6s} {rec['split']:8s} {rec['variant']:9s} L={rec['layers']} "
                  f"{rec['graph']:9s} seed={rec['seed']} n={rec.get('n_train', '-')} mae={rec['damage_mae']:.4f} "
                  f"({rec['seconds']}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
