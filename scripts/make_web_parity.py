"""Write the test cases that tests/web/check_parity.mjs uses to check the JavaScript copy of the engine.

    python scripts/make_web_parity.py                     # tests/web/parity_cases.json (small, committed)
    python scripts/make_web_parity.py --n 1500 --out big.json --seed 5     # bigger local check (1,515 cases)
    node tests/web/check_parity.mjs big.json

Every case has a request (the JSON a game engine would send) and what the Python code gives for it: the physics
layer result, the contact geometry, the model inputs and the model outputs.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crashdamage import dataset  # noqa: E402
from crashdamage.export import Predictor  # noqa: E402
from crashdamage.obstacles import PRESET_NAMES  # noqa: E402
from crashdamage.physics import scenario_from_dict, simulate, result_to_dict  # noqa: E402
from crashdamage.train import predict_arrays  # noqa: E402
from crashdamage.vehicle import FACES, PART_NAMES  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def request_from_scenario(sc):
    """The full, explicit JSON of a Scenario (every material, thickness and obstacle field)."""
    ob = sc.obstacle
    return {
        "speed_ms": sc.speed, "face": sc.face, "offset": sc.offset, "angle_deg": sc.angle_deg,
        "vehicle": {"mass_kg": sc.vehicle.mass_kg,
                    "materials": {p.name: p.material for p in sc.vehicle.parts},
                    "thickness_mm": {p.name: p.thickness_mm for p in sc.vehicle.parts}},
        "obstacle": {"name": ob.name, "material": ob.material, "shape": ob.shape, "form": ob.form,
                     "width": ob.width, "depth": ob.depth, "height": ob.height, "wall_mm": ob.wall_mm,
                     "perimeter": ob.perimeter, "mass_kg": ob.mass_kg, "after": ob.after, "stroke": ob.stroke,
                     "z_base": ob.z_base, "fixed": ob.mass_kg is None},
    }


def partial_request(rng):
    """A short request like a person would write it: a preset plus a few overrides (this goes through the
    quirks of obstacle_from_dict, for example that derived fields keep the preset's mid-range values)."""
    name = PRESET_NAMES[int(rng.integers(len(PRESET_NAMES)))]
    ob = {"name": name}
    for key, lo, hi in (("width", 0.05, 4.0), ("height", 0.3, 3.0), ("depth", 0.03, 1.0), ("z_base", 0.0, 1.8),
                        ("mass_kg", 50.0, 4000.0), ("wall_mm", 1.0, 25.0), ("perimeter", 0.1, 3.0),
                        ("stroke", 0.005, 0.5)):
        if rng.random() < 0.25:
            ob[key] = float(rng.uniform(lo, hi))
    if rng.random() < 0.1:
        ob["fixed"] = True
    if rng.random() < 0.1:
        ob["width"] = 0                                    # 0 means "keep the preset"
    if rng.random() < 0.15:
        ob["material"] = str(rng.choice(["concrete", "steel", "wood", "foam"]))
    if rng.random() < 0.15:
        ob["shape"] = str(rng.choice(["flat", "round", "sharp"]))
    if rng.random() < 0.15:
        ob["form"] = str(rng.choice(["solid", "thin_wall", "slender", "foam"]))
    if rng.random() < 0.1:
        ob["after"] = str(rng.choice(["rigid", "break"]))
    veh = {"mass_kg": float(rng.uniform(800, 2100))}
    if rng.random() < 0.4:
        veh["thickness_scale"] = float(rng.uniform(0.7, 1.5))
    mats = ["mild_steel", "high_strength_steel", "ultra_high_strength_steel", "aluminium", "plastic", "glass"]
    picks = [str(n) for n in rng.choice(PART_NAMES, size=int(rng.integers(0, 6)), replace=False)]
    if picks:
        if rng.random() < 0.5:                             # list form, as a Unity JsonUtility object writes it
            veh["materials"] = [{"part": n, "material": str(rng.choice(mats))} for n in picks]
        else:
            veh["materials"] = {n: str(rng.choice(mats)) for n in picks}
    if rng.random() < 0.3:
        veh["thickness_mm"] = [{"part": n, "mm": float(rng.uniform(0.5, 6.0))} for n in picks[:3]]
    req = {"face": str(rng.choice(FACES)), "offset": float(rng.uniform(-1, 1)),
           "angle_deg": float(rng.uniform(-60, 60)), "vehicle": veh, "obstacle": ob}
    if rng.random() < 0.5:
        req["speed_kmh"] = float(rng.uniform(5, 160))
    else:
        req["speed_ms"] = float(rng.uniform(1, 45))
    return req


def lst(a):
    return np.asarray(a).tolist()


def expected_for(req, predictor, with_features):
    sc = scenario_from_dict(req)
    res = simulate(sc)
    geo = res.geo
    inp = dataset.scenario_inputs(sc, geo)
    nx = np.concatenate([inp["node_x"], inp["phys_node"]], axis=1)
    gx = np.concatenate([inp["glob_x"], inp["phys_glob"]])
    out = predict_arrays(predictor.bundle, nx[None], gx[None])
    case = {
        "physics": {
            "damage": lst(res.damage), "level": lst(res.level), "detached": lst(res.detached), "direct": lst(res.direct),
            "crush_depth": res.crush_depth, "intrusion": res.intrusion, "peak_force_kn": res.peak_force_kn,
            "e_diss_kj": res.e_diss_kj, "e_car_kj": res.e_car_kj, "e_obstacle_kj": res.e_obstacle_kj,
            "unabsorbed_kj": res.unabsorbed_kj, "obstacle_damage": res.obstacle_damage,
            "obstacle_level": res.obstacle_level, "obstacle_depth": res.obstacle_depth,
            "breakthrough": res.breakthrough, "through": res.through, "airbag_front": res.airbag_front,
            "airbag_side": res.airbag_side, "delta_v": res.delta_v, "obstacle_speed_after": res.obstacle_speed_after,
            "effects": res.effects, "capacity_kj": lst(res.capacity_kj),
        },
        "geo": {k: lst(geo[k]) for k in ("overlap", "depth_start", "stroke", "dist", "point")}
               | {k: float(geo[k]) for k in ("width", "centre", "lo", "hi", "z0", "z1", "arm")},
        "model": {"damage": lst(out["damage"][0]), "detach_prob": lst(out["detach_prob"][0]),
                  "reg": lst(out["reg"][0]), "flags_prob": lst(out["flags_prob"][0])},
        "physics_report": result_to_dict(sc, res),
        "model_report": predictor.predict(sc, engine="model"),
    }
    if with_features:
        case["features"] = {"node": lst(nx), "glob": lst(gx)}
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="number of random cases (examples are always added)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "gnn_demo.pt"))
    ap.add_argument("--features", type=int, default=6, help="store the model inputs for the first N cases")
    ap.add_argument("--out", default=os.path.join(ROOT, "tests", "web", "parity_cases.json"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    predictor = Predictor(args.model)
    requests = []
    ex_dir = os.path.join(ROOT, "examples")
    for fn in sorted(os.listdir(ex_dir)):
        with open(os.path.join(ex_dir, fn)) as f:
            requests.append(json.load(f))
    for i in range(args.n):
        if i % 3 == 2:
            requests.append(partial_request(rng))
        else:
            requests.append(request_from_scenario(dataset.random_scenario(rng)))
    # corners: object passes over the car, extreme offsets and angles, very wide objects, a free object, speed 0
    base = {"face": "front", "vehicle": {"mass_kg": 1100}, "obstacle": {"name": "concrete_wall"}}
    for extra in ({"obstacle": {"name": "steel_pole", "z_base": 3.0}, "speed_kmh": 60},
                  {"offset": 1.0, "angle_deg": 40, "speed_kmh": 90},
                  {"offset": -1.0, "angle_deg": -40, "speed_kmh": 30, "face": "rear"},
                  {"obstacle": {"name": "wood_fence", "width": 9.0}, "face": "left", "speed_kmh": 70},
                  {"obstacle": {"name": "concrete_block", "mass_kg": 120}, "speed_kmh": 130},
                  {"obstacle": {"name": "steel_edge"}, "speed_kmh": 140, "face": "right", "offset": 0.3},
                  {"speed_kmh": 0}, {"speed_ms": 0, "speed_kmh": 25}):
        requests.append({**base, **extra})

    cases = []
    for i, req in enumerate(requests):
        cases.append({"request": req, "expected": expected_for(req, predictor, i < args.features)})

    # error function values for the GELU in the JavaScript network
    xs = [x / 100.0 for x in range(-800, 801, 7)]
    out = {"cases": cases, "erf": {"x": xs, "y": [math.erf(x) for x in xs]}, "seed": args.seed}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"wrote {len(cases)} cases to {args.out} ({os.path.getsize(args.out) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
