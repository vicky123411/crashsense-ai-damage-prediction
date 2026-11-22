"""Turn a scenario into numbers for the learning models.

Only things that are known BEFORE the crash is solved are used here: materials,
thicknesses, layout, where the obstacle touches, speed, masses. Nothing that comes
out of the energy solver is fed to the models.
"""
import math
import numpy as np

from .obstacles import OBSTACLE_FEATURE_NAMES, obstacle_features
from .physics import MIN_OVERLAP, contact_geometry, impact_energy, part_capacity_kj
from .vehicle import CABIN_DEPTH, EXTENT, FACE_HALF_SPAN, FACE_NORMAL, FACES, PART_FEATURE_NAMES, part_static_features

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


# -- impact frame ------------------------------------------------------------------------
# The same information as above, but measured from the point of impact instead of from the
# car's body axes: "how deep behind the hit point is this part" rather than "where is it on the
# car". A rear hit then looks to the model like a front hit on a car with different parts,
# so a side that never appeared in training is no longer a brand-new input.

IMPACT_PART_NAMES = ["depth_from_hit", "lateral_from_hit", "cz", "size_push", "size_across", "lz",
                     "section_push", "section_across", "pz"]
IMPACT_NODE_NAMES = ([n for n in PART_FEATURE_NAMES if n not in
                      ("cx", "cy", "cz", "lx", "ly", "lz", "px", "py", "pz")] + IMPACT_PART_NAMES + NODE_DYNAMIC_NAMES)
IMPACT_GLOBAL_NAMES = (["cabin_depth", "extent_push", "face_half_span", "speed", "speed_normal", "angle_deg",
                        "offset", "patch_width", "patch_centre", "patch_z0", "patch_z1", "arm", "log_vehicle_mass"]
                       + OBSTACLE_FEATURE_NAMES)
_BODY_POSE = [PART_FEATURE_NAMES.index(n) for n in ("cx", "cy", "cz", "lx", "ly", "lz", "px", "py", "pz")]
_KEEP = [i for i in range(len(PART_FEATURE_NAMES)) if i not in _BODY_POSE]


def _impact_axes(face):
    n = np.array(FACE_NORMAL[face], dtype=float)      # outward normal of the struck face (plan view)
    push = -n                                          # direction the obstacle pushes into the car
    lateral = np.array([-push[1], push[0]])            # push turned 90 degrees to the left
    return push, lateral


def node_features_impact(sc, geo=None):
    geo = geo or contact_geometry(sc)
    parts = sc.vehicle.parts
    static = np.array([part_static_features(p) for p in parts], dtype=np.float64)[:, _KEEP]
    push, lateral = _impact_axes(sc.face)
    centers = np.array([p.center for p in parts], dtype=float)
    sizes = np.array([p.size for p in parts], dtype=float)
    sect = np.array([p.section for p in parts], dtype=float)
    rel = centers[:, :2] - geo["point"][:2]
    ax, along = geo["ax"], geo["along"]
    pose = np.stack([rel @ push, rel @ lateral, centers[:, 2], sizes[:, ax], sizes[:, along], sizes[:, 2],
                     sect[:, ax], sect[:, along], sect[:, 2]], axis=1)
    dyn = np.stack([geo["overlap"], geo["depth_start"], geo["stroke"], geo["dist"],
                    (geo["overlap"] > MIN_OVERLAP).astype(float)], axis=1)
    return np.concatenate([static, pose, dyn], axis=1).astype(np.float32)


def global_features_impact(sc, geo=None):
    geo = geo or contact_geometry(sc)
    v_n = sc.speed * math.cos(math.radians(sc.angle_deg))
    scen = [CABIN_DEPTH[sc.face], EXTENT[sc.face], FACE_HALF_SPAN[sc.face], sc.speed, v_n, sc.angle_deg, sc.offset,
            geo["width"], geo["centre"], geo["z0"], geo["z1"], geo["arm"], math.log10(sc.vehicle.mass_kg)]
    return np.array(scen + obstacle_features(sc.obstacle), dtype=np.float32)
