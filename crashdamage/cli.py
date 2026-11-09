"""Command line entry point:  python -m crashdamage <command> ..."""
import argparse
import json
import os
import sys

from . import __version__


def _load_or_make(args):
    from . import dataset
    if getattr(args, "data", None) and os.path.exists(args.data):
        return dataset.load(args.data)
    data = dataset.generate(args.n, seed=args.seed)
    if getattr(args, "data", None):
        os.makedirs(os.path.dirname(os.path.abspath(args.data)), exist_ok=True)
        dataset.save(args.data, data)
    return data


def cmd_generate(args):
    from . import dataset
    data = dataset.generate(args.n, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    dataset.save(args.out, data)
    print(f"wrote {args.n} labelled scenarios to {args.out}")


def cmd_train(args):
    from . import dataset
    from .metrics import evaluate_predictions
    from .train import TrainConfig, predict, save_bundle, train
    data = _load_or_make(args)
    split = dataset.make_split(data, args.split, seed=args.seed)
    cfg = TrainConfig(kind=args.model, epochs=args.epochs, hidden=args.hidden, layers=args.layers, seed=args.seed,
                      physics_features=args.physics_features, energy_loss=args.energy_loss)
    bundle, hist = train(data, split, cfg, verbose=True)
    metrics = evaluate_predictions(data, split[2], predict(bundle, data, split[2]))
    print(json.dumps({k: round(v, 4) for k, v in metrics.items()}, indent=1))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_bundle(args.out, bundle)
    print(f"saved model to {args.out}")


def cmd_benchmark(args):
    from . import evaluate
    data = _load_or_make(args)
    variants = args.models.split(",") if args.models else list(evaluate.VARIANT_INFO)
    splits = args.splits.split(",") if args.splits else evaluate.SPLITS
    os.makedirs(args.out, exist_ok=True)
    raw = os.path.join(args.out, "benchmark_raw.json")
    results = evaluate.run_benchmark(data, variants, splits, seeds=tuple(range(args.seeds)), epochs=args.epochs,
                                     out_path=raw)
    lines = [f"# Benchmark ({len(data['speed'])} scenarios, {args.seeds} seeds, {args.epochs} epochs max)", ""]
    for sk in splits:
        lines += [f"## split: {sk}", "", evaluate.markdown_table(results, sk), ""]
    path = os.path.join(args.out, "benchmark.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {raw} and {path}")


def cmd_simulate(args):
    from .export import Predictor
    with open(args.scenario) as f:
        scenario = json.load(f)
    report = Predictor(args.model).predict(scenario, engine=args.engine)
    print(json.dumps(report, indent=2))


def cmd_serve(args):
    from .export import Predictor
    from .serve import serve
    serve(Predictor(args.model), host=args.host, port=args.port)


def cmd_import_fe(args):
    from . import dataset
    from .external import load_fe_dataset
    data = load_fe_dataset(args.cases, args.parts)
    dataset.save(args.out, data)
    print(f"wrote {len(data['speed'])} cases to {args.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="crashdamage", description="Vehicle-obstacle collision damage prediction")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="make a labelled dataset with the physics layer")
    g.add_argument("--n", type=int, default=8000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--out", default="data/synthetic.npz")
    g.set_defaults(fn=cmd_generate)

    t = sub.add_parser("train", help="train one model")
    t.add_argument("--data", default="data/synthetic.npz")
    t.add_argument("--n", type=int, default=8000, help="scenarios to generate if the data file does not exist")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--model", choices=["gnn", "node_mlp", "mlp"], default="gnn")
    t.add_argument("--split", choices=["random", "speed", "obstacle", "face"], default="random")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--hidden", type=int, default=96)
    t.add_argument("--layers", type=int, default=3)
    t.add_argument("--physics-features", action="store_true")
    t.add_argument("--energy-loss", type=float, default=0.0)
    t.add_argument("--out", default="models/gnn.pt")
    t.set_defaults(fn=cmd_train)

    b = sub.add_parser("benchmark", help="compare all models on all split types")
    b.add_argument("--data", default="data/synthetic.npz")
    b.add_argument("--n", type=int, default=8000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--seeds", type=int, default=3)
    b.add_argument("--epochs", type=int, default=40)
    b.add_argument("--models", default="", help="comma separated, default all")
    b.add_argument("--splits", default="", help="comma separated, default all")
    b.add_argument("--out", default="results")
    b.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("simulate", help="damage report for one scenario file")
    s.add_argument("scenario")
    s.add_argument("--model", default=None, help="trained model file; without it the physics layer is used")
    s.add_argument("--engine", choices=["auto", "physics", "model"], default="auto")
    s.set_defaults(fn=cmd_simulate)

    v = sub.add_parser("serve", help="small HTTP server for the game engine")
    v.add_argument("--model", default=None)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8765)
    v.set_defaults(fn=cmd_serve)

    f = sub.add_parser("import-fe", help="turn finite-element results (CSV) into a training set")
    f.add_argument("--cases", required=True)
    f.add_argument("--parts", required=True)
    f.add_argument("--out", default="data/fe.npz")
    f.set_defaults(fn=cmd_import_fe)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
