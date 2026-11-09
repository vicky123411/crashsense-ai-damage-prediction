"""Physics layer: a transparent, simplified energy model of a collision.

Idea in short
-------------
1. The relative speed, the mass of the car and the obstacle and the place of the
   hit give the energy that has to be dissipated (rigid-body impact, nearly plastic).
2. The contact patch (where the obstacle touches the car) decides which car parts
   are "in the line of fire", and in which order they are reached.
3. Each part resists with a crush force (strength x thickness x loaded perimeter).
   Parts at the same depth add up, parts behind each other come one after another.
4. The car and the obstacle push on each other with the same force, so the weaker
   one gives way first. This gives the damage of BOTH bodies.
5. Damage is spread to neighbouring parts through the part graph (door -> pillar
   -> windscreen ...).

This is a visual-plausibility model. It is not a validated crash-engineering model.
Every constant below is a modelling choice, listed in docs/method.md.
"""
from dataclasses import dataclass, field
import math
import numpy as np

from .materials import MATERIALS
from .obstacles import FOLD, Obstacle, obstacle_from_dict
from .vehicle import (CABIN_DEPTH, CAR_HEIGHT, EXTENT, FACE_HALF_SPAN, FACE_PLANE, FACES, N_PARTS,
                      PART_NAMES, Vehicle, adjacency_matrix, default_vehicle, ductility_factor)

DENS = 8.0                 # force multiplier once a solid block (engine, floor) has packed
F_BLOCK_CAP = 1.5e6        # N, upper limit of that packed-solid force
BLOCK_EFF = 0.15           # parts that can crush less than this share of their length act as solid blocks
INDIRECT_GAIN = 0.8        # damage given to a part that only receives force through its neighbours
RESTITUTION = 0.05         # nearly plastic impact
RIGID_FORCE = 1e9          # N, "cannot be pushed any further"
BIG_DEPTH = 20.0           # m, end of the obstacle's last depth segment
MIN_OVERLAP = 0.02         # parts covered less than this by the contact patch are not loaded directly
LEVEL_EDGES = [0.05, 0.2, 0.5, 0.85]
LEVEL_NAMES = ["intact", "scratched", "dented", "crushed", "destroyed"]
ELASTIC_FLOOR = 0.1        # a loaded part that does not yield still shows light marks
PROPAGATION_HOPS = 2
HOP_DECAY = 0.6
DETACH_MIN_DAMAGE = 0.3
SHAPE_CUT = {"flat": 1.0, "round": 0.9, "sharp": 0.7}   # sharper objects tear sheet material at a lower force
SHEET_MM = 2.5             # parts thinner than this are cut open instead of folded
AIRBAG_FRONT_DV = 5.5      # m/s, assumed trigger levels
AIRBAG_SIDE_DV = 4.0


@dataclass
class Scenario:
    vehicle: Vehicle
    obstacle: Obstacle
    face: str = "front"         # which side of the car is hit
    offset: float = 0.0         # -1..1, where along that side (0 = middle)
    angle_deg: float = 0.0      # approach angle measured from the face normal
    speed: float = 15.0         # m/s, relative (closing) speed

    def __post_init__(self):
        if self.face not in FACES:
            raise ValueError(f"face must be one of {FACES}")
        if not -1.0 <= self.offset <= 1.0:
            raise ValueError("offset must be between -1 and 1")
        if self.speed < 0:
            raise ValueError("speed must be >= 0")


@dataclass
class Result:
    damage: np.ndarray          # (N,) 0..1
    level: np.ndarray           # (N,) 0..4
    detached: np.ndarray        # (N,) bool
    direct: np.ndarray          # (N,) bool, part was in the contact patch
    crush_depth: float
    intrusion: float
    peak_force_kn: float
    e_diss_kj: float
    e_car_kj: float
    e_obstacle_kj: float
    obstacle_damage: float
    obstacle_level: int
    obstacle_depth: float
    breakthrough: bool
    airbag_front: bool
    airbag_side: bool
    delta_v: float
    obstacle_speed_after: float
    effects: dict = field(default_factory=dict)
    capacity_kj: np.ndarray = None   # (N,) energy each loaded part can take before it is fully crushed
    geo: dict = None                 # contact geometry used for this result
    through: bool = False            # the obstacle went through the whole load line of the car
    unabsorbed_kj: float = 0.0       # energy the modelled structures could not take (stays as motion)


def _axes(face):
    ax = 0 if face in ("front", "rear") else 1           # axis the car is pushed along
    sign = 1.0 if face in ("front", "left") else -1.0
    return ax, 1 - ax, sign


def contact_geometry(sc):
    """Everything that follows from the layout alone (no energy involved)."""
    v, ob, face = sc.vehicle, sc.obstacle, sc.face
    ax, along, sign = _axes(face)
    half = FACE_HALF_SPAN[face]
    width = min(ob.width, 2 * half)
    c = sc.offset * (half - width / 2)
    lo, hi = c - width / 2, c + width / 2
    z0, z1 = max(0.0, ob.z_base), min(CAR_HEIGHT, ob.z_base + ob.height)
    if z1 <= z0:                      # object passes over the car; touch the roof line
        z0, z1 = CAR_HEIGHT - 0.05, CAR_HEIGHT

    centers = np.array([p.center for p in v.parts], dtype=float)
    sizes = np.array([p.size for p in v.parts], dtype=float)
    eff = np.array([p.crush_eff for p in v.parts], dtype=float)

    q_max = sign * centers[:, ax] + sizes[:, ax] / 2
    depth_start = np.maximum(0.0, FACE_PLANE[face] - q_max)
    stroke = sizes[:, ax] * eff

    a0 = centers[:, along] - sizes[:, along] / 2
    a1 = centers[:, along] + sizes[:, along] / 2
    ov_a = np.clip(np.minimum(hi, a1) - np.maximum(lo, a0), 0, None) / sizes[:, along]
    zp0 = centers[:, 2] - sizes[:, 2] / 2
    zp1 = centers[:, 2] + sizes[:, 2] / 2
    ov_z = np.clip(np.minimum(z1, zp1) - np.maximum(z0, zp0), 0, None) / sizes[:, 2]
    overlap = np.clip(ov_a * ov_z, 0, 1)

    # contact point in plan view (x, y) and the arm to the centre of mass
    point = np.zeros(3)
    point[ax] = sign * FACE_PLANE[face]
    point[along] = c
    point[2] = 0.5 * (z0 + z1)
    dist = np.linalg.norm(centers - point, axis=1)
    return dict(ax=ax, along=along, sign=sign, width=width, centre=c, lo=lo, hi=hi, z0=z0, z1=z1,
                overlap=overlap, depth_start=depth_start, stroke=stroke, dist=dist,
                arm=abs(c), point=point)


def _car_forces(sc, geo):
    """Force of every part in every depth segment."""
    parts = sc.vehicle.parts
    ax = geo["ax"]
    sigma = np.array([MATERIALS[p.material].strength_mpa * 1e6 for p in parts])
    t = np.array([p.thickness_mm * 1e-3 for p in parts])
    sect = np.array([p.section[ax] for p in parts])
    cut = np.where(t * 1e3 < SHEET_MM, SHAPE_CUT[sc.obstacle.shape], 1.0)   # the obstacle shape matters for sheet parts
    f_full = FOLD * sigma * t * sect * cut
    engaged = geo["overlap"] > MIN_OVERLAP
    f_part = np.where(engaged, f_full * geo["overlap"], 0.0)
    start = geo["depth_start"]
    end = start + geo["stroke"]
    max_depth = EXTENT[sc.face]            # nothing can be crushed deeper than the car is long (or wide)
    pts = np.concatenate([[0.0], start[engaged], end[engaged], [max_depth]])
    b = np.unique(np.clip(pts, 0.0, max_depth))
    mid = 0.5 * (b[:-1] + b[1:])[:, None]
    eff = np.array([p.crush_eff for p in parts])
    packed = np.where(eff < BLOCK_EFF, np.minimum(DENS * f_part, F_BLOCK_CAP), 0.0)   # soft parts are used up, blocks stop the car
    f = np.where(mid < start[None, :], 0.0, np.where(mid < end[None, :], f_part[None, :], packed[None, :]))
    return b, f, f_part, engaged


def part_capacity_kj(sc, geo=None):
    """Energy (kJ) each loaded part can take by crushing along its stroke, before it is used up."""
    geo = geo or contact_geometry(sc)
    _, _, f_part, engaged = _car_forces(sc, geo)
    return np.where(engaged, f_part * geo["stroke"], 0.0) / 1e3


def _obstacle_curve(sc, geo):
    ob = sc.obstacle
    f_o = ob.plateau_force(geo["width"], geo["z1"] - geo["z0"])
    after = RIGID_FORCE if ob.after == "rigid" else 0.0
    b = np.array([0.0, max(ob.stroke, 1e-6), BIG_DEPTH])
    return b, np.array([f_o, after]), f_o


def _solve(bc, fc, bo, fo_seg, e_diss):
    """March the two bodies into each other. The body with the lower current resistance gives way."""
    fc_tot = fc.sum(axis=1)
    ahead = np.cumsum((fc_tot > 0)[::-1])[::-1] > 0       # is there any resistance left at or behind this segment?
    e_p = np.zeros(fc.shape[1])
    f_max = np.zeros(fc.shape[1])
    kc = ko = 0
    dc = do = 0.0
    e_left, e_obs, peak = e_diss, 0.0, 0.0
    breakthrough = through = False
    for _ in range(10000):
        if e_left <= 1e-6:
            break
        f_car, f_obs = fc_tot[kc], fo_seg[ko]
        if f_obs <= 0.0:
            breakthrough = True
            break
        if f_car <= f_obs:
            if f_car <= 0 and not ahead[kc]:           # nothing left in the line of the obstacle: it goes through
                through = True
                break
            length = bc[kc + 1] - dc
            adv = length if f_car <= 0 else min(length, e_left / f_car)
            e_p += fc[kc] * adv
            f_max = np.maximum(f_max, fc[kc])
            peak = max(peak, f_car)
            e_left -= f_car * adv
            dc += adv
            if adv >= length - 1e-12 and e_left > 1e-6:
                if kc + 2 < len(bc):
                    kc += 1
                else:                                  # reached the far side of the car
                    through = True
                    break
        else:
            length = bo[ko + 1] - do
            adv = min(length, e_left / f_obs)
            e_obs += f_obs * adv
            e_left -= f_obs * adv
            do += adv
            f_max = np.maximum(f_max, fc[kc] * (f_obs / f_car))
            peak = max(peak, f_obs)
            if adv >= length - 1e-12 and ko + 2 < len(bo):
                ko += 1
    return dict(e_p=e_p, f_max=f_max, dc=dc, do=do, e_obs=e_obs, peak=peak, e_left=max(e_left, 0.0),
                breakthrough=breakthrough, through=through)


def impact_energy(sc, geo=None):
    """Energy (J) that is dissipated, and the velocity change of the car."""
    geo = geo or contact_geometry(sc)
    m_c = sc.vehicle.mass_kg
    inv_eff_car = 1.0 / m_c + geo["arm"] ** 2 / sc.vehicle.yaw_inertia
    inv_mu = inv_eff_car + (0.0 if sc.obstacle.fixed else 1.0 / sc.obstacle.mass_kg)
    mu = 1.0 / inv_mu
    v_n = sc.speed * math.cos(math.radians(sc.angle_deg))
    e_diss = 0.5 * mu * v_n ** 2 * (1 - RESTITUTION ** 2)
    impulse = (1 + RESTITUTION) * mu * v_n
    dv = impulse / m_c
    v_obs = 0.0 if sc.obstacle.fixed else impulse / sc.obstacle.mass_kg
    return e_diss, dv, v_obs, v_n


def _level(d):
    return int(np.searchsorted(LEVEL_EDGES, d, side="right"))


def simulate(sc):
    geo = contact_geometry(sc)
    e_diss, dv, v_obs, v_n = impact_energy(sc, geo)
    bc, fc, f_part, engaged = _car_forces(sc, geo)
    bo, fo_seg, f_o = _obstacle_curve(sc, geo)
    sol = _solve(bc, fc, bo, fo_seg, e_diss)

    stroke = geo["stroke"]
    cap_e = np.where(engaged, f_part * stroke, 0.0)          # energy each loaded part can take (J)
    plastic = np.divide(sol["e_p"], cap_e, out=np.zeros_like(cap_e), where=cap_e > 0)
    elastic = ELASTIC_FLOOR * np.divide(np.minimum(sol["f_max"], f_part), f_part,
                                        out=np.zeros_like(f_part), where=f_part > 0)
    direct = engaged & (sol["f_max"] > 0)
    d_dir = np.where(direct, np.maximum(plastic, elastic), 0.0).clip(0, 1)
    f_dir = np.where(direct, sol["f_max"], 0.0)

    # Force travels through the part graph. A part cannot pass on more than it can carry,
    # and a part that only receives force through its neighbours is damaged in proportion.
    a = adjacency_matrix()
    parts = sc.vehicle.parts
    sigma = np.array([MATERIALS[p.material].strength_mpa * 1e6 for p in parts])
    thick = np.array([p.thickness_mm * 1e-3 for p in parts])
    f_yield = FOLD * sigma * thick * np.array([np.mean(p.section) for p in parts])
    g = np.array([ductility_factor(p.material) for p in parts])
    f_in = np.zeros(N_PARTS)
    d_in = np.zeros(N_PARTS)
    for hop in range(PROPAGATION_HOPS):
        decay = 1.0 if hop == 0 else HOP_DECAY
        carried = np.minimum(np.maximum(f_dir, f_in), f_yield)           # a part passes on no more than it carries
        f_in = np.maximum(f_in, (a * carried[:, None]).max(axis=0) * decay)
        nb_cap = (a * np.maximum(d_dir, d_in)[:, None]).max(axis=0) * decay  # never worse than the neighbour it hangs on
        d_new = np.minimum(INDIRECT_GAIN * np.minimum(1.0, f_in / f_yield) / g, nb_cap)
        d_in = np.maximum(d_in, np.minimum(d_new, 1.0))
    d = np.clip(np.maximum(d_dir, d_in), 0, 1)
    d = np.where(d < 0.01, 0.0, d)
    f = np.maximum(f_dir, f_in)

    attach = np.array([p.attach_kn * 1e3 for p in sc.vehicle.parts])
    detached = (f >= attach) & (d >= DETACH_MIN_DAMAGE)
    level = np.array([_level(x) for x in d])

    # obstacle
    cap_o = f_o * sc.obstacle.stroke
    d_o = min(1.0, sol["e_obs"] / cap_o) if cap_o > 0 else 0.0
    if sol["breakthrough"] and sc.obstacle.after == "break":
        d_o = 1.0
    d_o = float(d_o)

    depth = float(sol["dc"])
    intrusion = max(0.0, depth - CABIN_DEPTH[sc.face])
    face_front = sc.face == "front" and abs(sc.angle_deg) <= 30
    side = sc.face in ("left", "right")
    airbag_front = bool(face_front and dv >= AIRBAG_FRONT_DV)
    airbag_side = bool(side and dv >= AIRBAG_SIDE_DV)

    res = Result(damage=d, level=level, detached=detached, direct=direct, crush_depth=depth,
                 intrusion=float(intrusion), peak_force_kn=float(sol["peak"] / 1e3), e_diss_kj=e_diss / 1e3,
                 e_car_kj=float(sol["e_p"].sum() / 1e3), e_obstacle_kj=float(sol["e_obs"] / 1e3),
                 obstacle_damage=d_o, obstacle_level=_level(d_o), obstacle_depth=float(sol["do"]),
                 breakthrough=bool(sol["breakthrough"]), airbag_front=airbag_front, airbag_side=airbag_side,
                 delta_v=float(dv), obstacle_speed_after=float(v_obs),
                 through=bool(sol["through"]), unabsorbed_kj=float(sol["e_left"] / 1e3))
    res.effects = effect_hints(sc, res)
    res.capacity_kj = cap_e / 1e3
    res.geo = geo
    return res


def effect_hints(sc, res):
    """Simple rules that tell the game engine which effects make sense."""
    idx = {n: i for i, n in enumerate(PART_NAMES)}
    kinds = [p.kind for p in sc.vehicle.parts]
    mats = [p.material for p in sc.vehicle.parts]
    metal = {"mild_steel", "high_strength_steel", "ultra_high_strength_steel", "aluminium"}
    hard = sc.obstacle.material in ("concrete", "steel")
    metal_hit = any(res.direct[i] and mats[i] in metal and res.level[i] >= 2 for i in range(N_PARTS))
    glass_break = any(kinds[i] == "glass" and res.level[i] >= 3 for i in range(N_PARTS))
    leak = any(n in idx and res.level[idx[n]] >= 3 for n in ("radiator", "engine"))
    broken = bool(res.detached.any() or (res.level >= 4).any() or res.obstacle_damage >= 1.0 or res.through)
    return {
        "sparks": bool(hard and metal_hit and res.e_diss_kj > 5.0),
        "glass_shards": bool(glass_break),
        "smoke": bool(leak),
        "fluid_leak": bool(leak),
        "debris": broken,
        "dust": bool(sc.obstacle.material in ("concrete", "wood") and res.obstacle_level >= 2),
        "airbag_front": res.airbag_front,
        "airbag_side": res.airbag_side,
    }


def result_to_dict(sc, res):
    parts = [{"id": PART_NAMES[i], "damage": round(float(res.damage[i]), 4), "level": int(res.level[i]),
              "level_name": LEVEL_NAMES[int(res.level[i])], "detached": bool(res.detached[i]),
              "direct": bool(res.direct[i])} for i in range(N_PARTS)]
    return {
        "parts": parts,
        "obstacle": {"name": sc.obstacle.name, "material": sc.obstacle.material,
                     "damage": round(res.obstacle_damage, 4), "level": res.obstacle_level,
                     "level_name": LEVEL_NAMES[res.obstacle_level], "deformation_m": round(res.obstacle_depth, 4),
                     "broken_through": res.breakthrough, "speed_after_ms": round(res.obstacle_speed_after, 3)},
        "summary": {"crush_depth_m": round(res.crush_depth, 4), "cabin_intrusion_m": round(res.intrusion, 4),
                    "peak_force_kn": round(res.peak_force_kn, 2), "energy_total_kj": round(res.e_diss_kj, 2),
                    "energy_car_kj": round(res.e_car_kj, 2), "energy_obstacle_kj": round(res.e_obstacle_kj, 2),
                    "energy_unabsorbed_kj": round(res.unabsorbed_kj, 2), "went_through_car": res.through,
                    "delta_v_ms": round(res.delta_v, 3)},
        "effects": res.effects,
        "impact": {"face": sc.face, "offset": sc.offset, "angle_deg": sc.angle_deg,
                   "speed_ms": sc.speed, "speed_kmh": round(sc.speed * 3.6, 1)},
    }


def _pairs(x, value_key):
    """Accept {"door_l": value} or [{"part": "door_l", value_key: value}], the form a Unity
    JsonUtility object can write."""
    if isinstance(x, dict):
        return dict(x)
    return {item["part"]: item[value_key] for item in x}


def scenario_from_dict(d):
    """Build a Scenario from the JSON the game engine sends."""
    from dataclasses import replace
    vd = d.get("vehicle", {})
    veh = default_vehicle(float(vd.get("mass_kg", 1100.0)) or 1100.0)
    scale = float(vd.get("thickness_scale", 1.0)) or 1.0
    mats = _pairs(vd.get("materials", {}), "material")
    thick = _pairs(vd.get("thickness_mm", {}), "mm")
    parts = []
    for p in veh.parts:
        m = mats.get(p.name, p.material)
        if m not in MATERIALS:
            raise ValueError(f"unknown material {m!r} for part {p.name}")
        t = float(thick.get(p.name, p.thickness_mm * scale))
        parts.append(replace(p, material=m, thickness_mm=t))
    veh.parts = parts
    ob = obstacle_from_dict(d.get("obstacle", {"name": "concrete_wall"}))
    if float(d.get("speed_ms") or 0.0) > 0:          # 0 means "not given" (see obstacle_from_dict)
        speed = float(d["speed_ms"])
    else:
        speed = float(d.get("speed_kmh", 50.0)) / 3.6
    return Scenario(veh, ob, face=d.get("face", "front"), offset=float(d.get("offset", 0.0)),
                    angle_deg=float(d.get("angle_deg", 0.0)), speed=speed)
