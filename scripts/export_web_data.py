"""Write the data files that the web demo in docs/ reads.

    python scripts/export_web_data.py --model models/gnn_demo.pt --metrics results/demo_model_metrics.txt

docs/data/static.json   the car (28 parts, part graph), materials, obstacle presets, constants, example scenarios
docs/data/model.json    settings of the trained network and where each weight sits in model.bin
docs/data/model.bin     all weights, normaliser statistics and the part graph as little-endian float32

The browser code in docs/js/ is a JavaScript copy of the physics layer and of the network; the numbers it
gives are checked against Python by tests/web/check_parity.mjs.
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crashdamage import materials as M, obstacles as O, physics as P, vehicle as V  # noqa: E402
from crashdamage.dataset import FACE_PROBS  # noqa: E402


def static_tables(examples_dir):
    parts = []
    for p in V.PART_TEMPLATE:
        parts.append(dict(name=p.name, kind=p.kind, material=p.material, thickness_mm=p.thickness_mm,
                          center=list(p.center), size=list(p.size), crush_eff=p.crush_eff, section=list(p.section),
                          attach_kn=None if np.isinf(p.attach_kn) else p.attach_kn, mass_kg=p.mass_kg,
                          options=list(p.options)))
    # the part graph uses float32 weights in Python, so the same rounded values are written here
    adj = V.adjacency_matrix()
    edges = [[i, j, float(adj[i, j])] for i, j, _ in V.EDGES]

    presets = {}
    for name, spec in O.PRESETS.items():
        ob = O.default_obstacle(name)
        presets[name] = dict(
            spec={k: (list(v) if isinstance(v, tuple) else v) for k, v in spec.items()},
            default=dict(name=ob.name, material=ob.material, shape=ob.shape, form=ob.form, width=ob.width,
                         depth=ob.depth, height=ob.height, wall_mm=ob.wall_mm, perimeter=ob.perimeter,
                         mass_kg=ob.mass_kg, after=ob.after, stroke=ob.stroke, z_base=ob.z_base))

    examples = []
    for fn in sorted(os.listdir(examples_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(examples_dir, fn)) as f:
                req = json.load(f)
            examples.append(dict(id=fn[:-5], comment=req.pop("_comment", ""), request=req))

    return dict(
        constants=dict(
            DENS=P.DENS, F_BLOCK_CAP=P.F_BLOCK_CAP, BLOCK_EFF=P.BLOCK_EFF, INDIRECT_GAIN=P.INDIRECT_GAIN,
            RESTITUTION=P.RESTITUTION, RIGID_FORCE=P.RIGID_FORCE, BIG_DEPTH=P.BIG_DEPTH, MIN_OVERLAP=P.MIN_OVERLAP,
            ELASTIC_FLOOR=P.ELASTIC_FLOOR, PROPAGATION_HOPS=P.PROPAGATION_HOPS, HOP_DECAY=P.HOP_DECAY,
            DETACH_MIN_DAMAGE=P.DETACH_MIN_DAMAGE, SHEET_MM=P.SHEET_MM, AIRBAG_FRONT_DV=P.AIRBAG_FRONT_DV,
            AIRBAG_SIDE_DV=P.AIRBAG_SIDE_DV, FOLD=O.FOLD, SOLID_K=O.SOLID_K, SLENDER_K=O.SLENDER_K,
            CAR_LENGTH=V.CAR_LENGTH, CAR_WIDTH=V.CAR_WIDTH, CAR_HEIGHT=V.CAR_HEIGHT),
        shape_cut=P.SHAPE_CUT,
        level_edges=P.LEVEL_EDGES, level_names=P.LEVEL_NAMES,
        faces=V.FACES, face_plane=V.FACE_PLANE, face_half_span=V.FACE_HALF_SPAN, cabin_depth=V.CABIN_DEPTH,
        extent=V.EXTENT, kinds=V.KINDS, shapes=O.SHAPES, forms=O.FORMS, after_modes=O.AFTER,
        materials={n: dict(density=m.density, youngs_gpa=m.youngs_gpa, strength_mpa=m.strength_mpa,
                           fracture_strain=m.fracture_strain, brittle=m.brittle) for n, m in M.MATERIALS.items()},
        vehicle_materials=["mild_steel", "high_strength_steel", "ultra_high_strength_steel", "aluminium", "plastic",
                           "glass"],
        obstacle_materials=["concrete", "steel", "wood", "foam"],
        parts=parts, edges=edges, obstacle_presets=presets, examples=examples,
        # what the random scenarios of the training set covered (dataset.py, vehicle.py)
        training=dict(speed_ms=[4.0, 36.0], mass_kg=[900.0, 1900.0], thickness_scale=[0.9, 1.2],
                      thickness_jitter=[0.95, 1.05], offset=[-1.0, 1.0], angle_deg=[-40.0, 40.0],
                      face_probs=dict(zip(V.FACES, FACE_PROBS))),
    )


def read_metrics(path):
    """The JSON block that `python -m crashdamage train` prints after training."""
    if not path or not os.path.exists(path):
        return None
    text = open(path).read()
    blocks = re.findall(r"\{[^{}]*\}", text, flags=re.S)
    return json.loads(blocks[-1]) if blocks else None


def export_model(model_path, out_dir, metrics):
    ck = torch.load(model_path, map_location="cpu", weights_only=True)
    state = ck["state"]
    tensors = []
    chunks = []
    offset = 0

    def add(name, arr):
        nonlocal offset
        a = np.ascontiguousarray(np.asarray(arr), dtype="<f4")
        tensors.append(dict(name=name, shape=list(a.shape), offset=offset))
        chunks.append(a.tobytes())
        offset += a.nbytes

    for k, v in state.items():
        add(k, v.numpy())
    for k, v in ck["norm"].items():
        add("norm." + k, v.numpy())
    add("reg_mean", ck["reg_mean"].numpy())
    add("reg_std", ck["reg_std"].numpy())
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model.bin"), "wb") as f:
        f.write(b"".join(chunks))
    cfg = ck["cfg"]
    manifest = dict(config=cfg, n_node_feat=int(ck["n_node_feat"]), n_glob_feat=int(ck["n_glob_feat"]),
                    tensors=tensors, bytes=offset, metrics=metrics,
                    note="Graph network trained on 8000 random scenarios labelled by the physics layer.")
    with open(os.path.join(out_dir, "model.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(root, "models", "gnn_demo.pt"))
    ap.add_argument("--metrics", default=os.path.join(root, "results", "demo_model_metrics.txt"))
    ap.add_argument("--examples", default=os.path.join(root, "examples"))
    ap.add_argument("--out", default=os.path.join(root, "docs", "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "static.json"), "w") as f:
        json.dump(static_tables(args.examples), f, indent=1)
    man = export_model(args.model, args.out, read_metrics(args.metrics))
    print(f"wrote {args.out}/static.json, model.json and model.bin ({man['bytes'] / 1e3:.0f} kB, "
          f"{len(man['tensors'])} tensors)")


if __name__ == "__main__":
    main()
