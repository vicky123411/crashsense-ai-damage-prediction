"""Turn a scenario into numbers for the learning models.

Only things that are known BEFORE the crash is solved are used here: materials,
thicknesses, layout, where the obstacle touches, speed, masses. Nothing that comes
out of the energy solver is fed to the models.
"""
import math
import numpy as np

from .obstacles import OBSTACLE_FEATURE_NAMES, obstacle_features
from .physics import MIN_OVERLAP, contact_geometry, impact_energy, part_capacity_kj
from .vehicle import FACES, PART_FEATURE_NAMES, part_static_features

NODE_DYNAMIC_NAMES = ["overlap", "depth_start", "stroke", "dist_to_contact", "in_contact_patch"]
NODE_FEATURE_NAMES = PART_FEATURE_NAMES + NODE_DYNAMIC_NAMES
GLOBAL_SCENARIO_NAMES = ([f"face_{f}" for f in FACES] +
                         ["speed", "speed_normal", "angle_deg", "offset", "patch_width", "patch_centre",
                          "patch_z0", "patch_z1", "arm", "log_vehicle_mass"])
GLOBAL_FEATURE_NAMES = GLOBAL_SCENARIO_NAMES + OBSTACLE_FEATURE_NAMES

N_NODE_FEATURES = len(NODE_FEATURE_NAMES)
N_GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)


def node_features(sc, geo=None):
    geo = geo or contact_geometry(sc)
    static = np.array([part_static_features(p) for p in sc.vehicle.parts], dtype=np.float32)
    dyn = np.stack([geo["overlap"], geo["depth_start"], geo["stroke"], geo["dist"],
                    (geo["overlap"] > MIN_OVERLAP).astype(float)], axis=1).astype(np.float32)
    return np.concatenate([static, dyn], axis=1)


def global_features(sc, geo=None):
    geo = geo or contact_geometry(sc)
    face = [1.0 if sc.face == f else 0.0 for f in FACES]
    v_n = sc.speed * math.cos(math.radians(sc.angle_deg))
    scen = face + [sc.speed, v_n, sc.angle_deg, sc.offset, geo["width"], geo["centre"], geo["z0"], geo["z1"],
                   geo["arm"], math.log10(sc.vehicle.mass_kg)]
    return np.array(scen + obstacle_features(sc.obstacle), dtype=np.float32)


PHYS_NODE_NAMES = ["log_capacity_kj", "log_energy_to_capacity"]
PHYS_GLOBAL_NAMES = ["log_e_diss_kj", "log_capacity_total_kj", "log_energy_ratio"]


def physics_features(sc, geo=None):
    """Optional extra inputs from closed-form physics: impact energy and the energy each loaded
    part could take. They need no solver run, only masses, speed and the part list.

    Returns (per-part array (N, 2), global array (3,)).
    """
    geo = geo or contact_geometry(sc)
    e_diss_kj = impact_energy(sc, geo)[0] / 1e3
    cap = part_capacity_kj(sc, geo)
    log_e = math.log10(max(e_diss_kj, 1e-3))
    log_cap_total = math.log10(max(float(cap.sum()), 1e-3))
    node = np.stack([np.log10(cap + 1e-3), np.log10(e_diss_kj / (cap + 1e-3) + 1e-3)], axis=1).astype(np.float32)
    glob = np.array([log_e, log_cap_total, log_e - log_cap_total], dtype=np.float32)
    return node, glob
