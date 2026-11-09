"""Bring in labels from a finite-element solver (OpenRadioss, LS-DYNA, ...).

The models never need to know where the labels come from. This module reads two CSV files
and produces the same arrays that ``dataset.generate`` makes from the physics layer.

cases.csv, one row per simulation
    case_id, face, offset, angle_deg, speed_kmh, vehicle_mass_kg, obstacle
      face      front | rear | left | right
      offset    -1..1, where along that side the obstacle touches (0 = middle)
      obstacle  name of a preset (see ``obstacles.PRESETS``)
    optional columns
      obstacle_width, obstacle_depth, obstacle_height, obstacle_wall_mm, obstacle_perimeter,
      obstacle_mass_kg, obstacle_z_base, obstacle_stroke      override the preset values
      mat:<part_id>, t_mm:<part_id>                           material name / thickness of a part
      crush_depth_m, intrusion_m, peak_force_kn, obstacle_damage, airbag_front, airbag_side, through
                                                              result labels (missing = 0)

parts.csv, one row per (simulation, part)
    case_id, part_id, damage, detached
      damage    0..1, how much of the part is used up. How you get this number from the solver
                (plastic strain over failure strain, crushed share of the part length, ...) is your
                choice and should be written down next to the dataset.
      detached  0 or 1

Parts that do not appear for a case are taken as undamaged.

This reader has been tested on files written by ``write_csv`` below. It has not been tried on real
solver output, so check the first few cases by hand.
"""
import csv
from collections import defaultdict

import numpy as np

from .dataset import scenario_inputs
from .metrics import level_from_damage
from .obstacles import PRESET_NAMES
from .physics import (MIN_OVERLAP, contact_geometry, impact_energy, part_capacity_kj, scenario_from_dict)
from .vehicle import FACES, N_PARTS, PART_INDEX, PART_NAMES

OBSTACLE_FIELDS = ["width", "depth", "height", "wall_mm", "perimeter", "mass_kg", "z_base", "stroke"]


def _num(row, key, default=0.0):
    v = row.get(key)
    return default if v in (None, "") else float(v)


def scenario_from_row(row):
    veh = {"mass_kg": _num(row, "vehicle_mass_kg", 1100.0), "materials": {}, "thickness_mm": {}}
    for key, value in row.items():
        if value in (None, ""):
            continue
        if key.startswith("mat:"):
            veh["materials"][key[4:]] = value
        elif key.startswith("t_mm:"):
            veh["thickness_mm"][key[5:]] = float(value)
    ob = {"name": row["obstacle"]}
    for f in OBSTACLE_FIELDS:
        v = row.get(f"obstacle_{f}")
        if v not in (None, ""):
            ob[f] = float(v)
    return scenario_from_dict({"face": row.get("face") or "front", "offset": _num(row, "offset"),
                               "angle_deg": _num(row, "angle_deg"), "speed_kmh": float(row["speed_kmh"]),
                               "vehicle": veh, "obstacle": ob})


def load_fe_dataset(cases_csv, parts_csv):
    with open(cases_csv, newline="") as f:
        cases = list(csv.DictReader(f))
    labels = defaultdict(dict)
    with open(parts_csv, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["part_id"]
            if pid not in PART_INDEX:
                raise ValueError(f"unknown part_id {pid!r} (case {row['case_id']}); known ids: {', '.join(PART_NAMES)}")
            labels[row["case_id"]][pid] = (float(row["damage"]), float(row.get("detached") or 0))
    out = {k: [] for k in ["node_x", "glob_x", "phys_node", "phys_glob", "damage", "level", "detached", "direct",
                           "ecap_kj", "e_diss_kj", "glob_reg", "obstacle_level", "flags", "speed", "face", "preset"]}
    for row in cases:
        cid = row["case_id"]
        if cid not in labels:
            raise ValueError(f"case {cid} has no rows in the parts file")
        sc = scenario_from_row(row)
        geo = contact_geometry(sc)
        for k, v in scenario_inputs(sc, geo).items():
            out[k].append(v)
        damage, detached = np.zeros(N_PARTS), np.zeros(N_PARTS)
        for pid, (dm, dt) in labels[cid].items():
            damage[PART_INDEX[pid]] = np.clip(dm, 0.0, 1.0)
            detached[PART_INDEX[pid]] = 1.0 if dt >= 0.5 else 0.0
        obstacle_damage = float(np.clip(_num(row, "obstacle_damage"), 0.0, 1.0))
        out["damage"].append(damage)
        out["level"].append(level_from_damage(damage))
        out["detached"].append(detached)
        out["direct"].append((geo["overlap"] > MIN_OVERLAP).astype(np.float32))
        out["ecap_kj"].append(part_capacity_kj(sc, geo))
        out["e_diss_kj"].append(impact_energy(sc, geo)[0] / 1e3)
        out["glob_reg"].append([_num(row, "crush_depth_m"), _num(row, "intrusion_m"),
                                _num(row, "peak_force_kn") / 1000.0, obstacle_damage])
        out["obstacle_level"].append(int(level_from_damage(obstacle_damage)))
        out["flags"].append([_num(row, "airbag_front"), _num(row, "airbag_side"), _num(row, "through")])
        out["speed"].append(sc.speed)
        out["face"].append(FACES.index(sc.face))
        out["preset"].append(PRESET_NAMES.index(sc.obstacle.name) if sc.obstacle.name in PRESET_NAMES else -1)
    data = {k: np.array(v) for k, v in out.items()}
    for k in ("node_x", "glob_x", "phys_node", "phys_glob", "damage", "detached", "direct", "ecap_kj", "glob_reg",
              "flags", "e_diss_kj", "speed"):
        data[k] = data[k].astype(np.float32)
    for k in ("level", "obstacle_level", "face", "preset"):
        data[k] = data[k].astype(np.int64)
    data["level"] = level_from_damage(data["damage"]).astype(np.int64)
    data["obstacle_level"] = level_from_damage(data["glob_reg"][:, 3]).astype(np.int64)
    data["source"] = np.array("external_fe")
    return data


def write_csv(scenarios, results, cases_path, parts_path):
    """Write scenarios and their results in the layout above (used for tests and as an example)."""
    mat_cols = [f"mat:{n}" for n in PART_NAMES]
    t_cols = [f"t_mm:{n}" for n in PART_NAMES]
    head = (["case_id", "face", "offset", "angle_deg", "speed_kmh", "vehicle_mass_kg", "obstacle"]
            + [f"obstacle_{f}" for f in OBSTACLE_FIELDS]
            + ["crush_depth_m", "intrusion_m", "peak_force_kn", "obstacle_damage", "airbag_front", "airbag_side",
               "through"] + mat_cols + t_cols)
    with open(cases_path, "w", newline="") as fc, open(parts_path, "w", newline="") as fp:
        wc, wp = csv.writer(fc), csv.writer(fp)
        wc.writerow(head)
        wp.writerow(["case_id", "part_id", "damage", "detached"])
        for i, (sc, res) in enumerate(zip(scenarios, results)):
            ob = sc.obstacle
            row = [i, sc.face, repr(sc.offset), repr(sc.angle_deg), repr(sc.speed * 3.6), repr(sc.vehicle.mass_kg), ob.name]
            row += ["" if getattr(ob, f) is None else repr(getattr(ob, f)) for f in OBSTACLE_FIELDS]
            row += [repr(res.crush_depth), repr(res.intrusion), repr(res.peak_force_kn), repr(res.obstacle_damage),
                    int(res.airbag_front), int(res.airbag_side), int(res.through)]
            row += [p.material for p in sc.vehicle.parts] + [repr(p.thickness_mm) for p in sc.vehicle.parts]
            wc.writerow(row)
            for j, name in enumerate(PART_NAMES):
                wp.writerow([i, name, repr(float(res.damage[j])), int(res.detached[j])])
