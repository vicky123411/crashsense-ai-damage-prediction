"""A simple compact-car model made of boxes.

The car is described in a body frame: x points forward, y to the left, z up, the
origin is on the ground below the middle of the car. Each part is an axis-aligned
box with a material, a thickness and a few strength numbers. Parts are joined in
a graph (edges = load paths and fixings) that is used both by the physics layer
and by the graph neural network.
"""
from dataclasses import dataclass, replace
import numpy as np

from .materials import MATERIALS, material_features

CAR_LENGTH = 4.05
CAR_WIDTH = 1.75
CAR_HEIGHT = 1.45

# outward direction of each face of the car (plan view) and where the face is
FACES = ["front", "rear", "left", "right"]
FACE_NORMAL = {"front": (1.0, 0.0), "rear": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
FACE_PLANE = {"front": 2.025, "rear": 2.025, "left": 0.875, "right": 0.875}
# half length of each face (the direction you slide along when you change the offset)
FACE_HALF_SPAN = {"front": 0.875, "rear": 0.875, "left": 2.025, "right": 2.025}
# depth (m) from the outer surface where the passenger cell starts
CABIN_DEPTH = {"front": 0.90, "rear": 0.80, "left": 0.36, "right": 0.36}
# size of the car along the push direction: nothing can be crushed deeper than this
EXTENT = {"front": CAR_LENGTH, "rear": CAR_LENGTH, "left": CAR_WIDTH, "right": CAR_WIDTH}

KINDS = ["structure", "panel", "glass", "mechanical", "wheel", "trim"]


@dataclass(frozen=True)
class Part:
    name: str
    kind: str
    material: str
    thickness_mm: float
    center: tuple            # x, y, z in metres
    size: tuple              # lx, ly, lz in metres
    crush_eff: float         # fraction of the box length that can be crushed before it packs solid
    section: tuple           # load-bearing perimeter (m) when loaded along x, y, z
    attach_kn: float         # force (kN) that tears the part off its mounts (inf = welded)
    mass_kg: float
    options: tuple = ()      # materials a car variant may use for this part


def _p(name, kind, material, t, center, size, eff, section, attach, mass, options=()):
    return Part(name, kind, material, float(t), center, size, eff, section,
                float(attach), float(mass), tuple(options) or (material,))


INF = float("inf")


def _template():
    p = []
    add = p.append
    add(_p("bumper_front", "trim", "plastic", 3.0, (1.95, 0, 0.52), (0.15, 1.70, 0.34), 0.5, (0.6, 0.3, 0.3), 5, 8))
    add(_p("beam_front", "structure", "high_strength_steel", 1.8, (1.84, 0, 0.50), (0.10, 1.50, 0.15), 0.6, (0.5, 0.5, 0.5), INF, 12,
           ("high_strength_steel", "aluminium", "ultra_high_strength_steel")))
    for s, y in (("l", 0.45), ("r", -0.45)):
        add(_p(f"rail_front_{s}", "structure", "high_strength_steel", 1.6, (1.29, y, 0.50), (1.02, 0.12, 0.15), 0.65, (0.5, 0.3, 0.3), INF, 10,
               ("high_strength_steel", "mild_steel")))
    add(_p("radiator", "mechanical", "aluminium", 0.4, (1.75, 0, 0.55), (0.08, 0.60, 0.50), 0.7, (0.4, 0.3, 0.3), 8, 6))
    add(_p("engine", "mechanical", "aluminium", 20.0, (1.075, 0, 0.55), (0.65, 0.60, 0.55), 0.05, (1.0, 0.8, 0.8), 60, 130))
    add(_p("bonnet", "panel", "mild_steel", 0.8, (1.25, 0, 0.93), (1.15, 1.30, 0.04), 0.5, (0.25, 0.15, 0.3), 15, 14,
           ("mild_steel", "aluminium")))
    for s, y in (("l", 0.80), ("r", -0.80)):
        add(_p(f"fender_{s}", "panel", "mild_steel", 0.7, (1.40, y, 0.80), (1.10, 0.10, 0.45), 0.5, (0.5, 0.4, 0.3), 45, 6,
               ("mild_steel", "aluminium")))
    add(_p("windscreen", "glass", "glass", 4.0, (0.62, 0, 1.12), (0.50, 1.30, 0.55), 0.05, (1.3, 0.5, 1.3), INF, 12))
    for s, y in (("l", 0.78), ("r", -0.78)):
        add(_p(f"pillar_a_{s}", "structure", "ultra_high_strength_steel", 1.6, (0.62, y, 1.05), (0.35, 0.10, 0.70), 0.5, (0.6, 0.5, 0.6), INF, 6,
               ("high_strength_steel", "ultra_high_strength_steel")))
    add(_p("firewall", "structure", "high_strength_steel", 1.0, (0.72, 0, 0.70), (0.05, 1.50, 0.70), 0.3, (3.0, 0.6, 1.5), INF, 15))
    for s, y in (("l", 0.81), ("r", -0.81)):
        add(_p(f"door_{s}", "panel", "high_strength_steel", 1.0, (0.15, y, 0.78), (0.95, 0.14, 0.62), 0.7, (0.6, 0.9, 0.5), 25, 22,
               ("mild_steel", "high_strength_steel")))
    for s, y in (("l", 0.80), ("r", -0.80)):
        add(_p(f"pillar_b_{s}", "structure", "ultra_high_strength_steel", 1.5, (-0.40, y, 0.80), (0.12, 0.12, 0.90), 0.5, (0.5, 0.5, 0.6), INF, 8,
               ("high_strength_steel", "ultra_high_strength_steel")))
    for s, y in (("l", 0.78), ("r", -0.78)):
        add(_p(f"sill_{s}", "structure", "high_strength_steel", 1.4, (0.0, y, 0.30), (2.40, 0.14, 0.18), 0.6, (0.6, 0.6, 0.4), INF, 14,
               ("mild_steel", "high_strength_steel")))
    add(_p("floor_cabin", "structure", "high_strength_steel", 1.2, (0.0, 0.0, 0.28), (2.40, 1.40, 0.12), 0.1, (2.0, 1.2, 4.0), INF, 60))
    add(_p("roof", "panel", "mild_steel", 0.8, (-0.20, 0, 1.40), (1.70, 1.45, 0.04), 0.5, (0.4, 0.3, 0.5), INF, 18,
           ("mild_steel", "high_strength_steel", "aluminium")))
    add(_p("rear_glass", "glass", "glass", 3.5, (-1.35, 0, 1.12), (0.25, 1.20, 0.55), 0.05, (1.2, 0.5, 1.2), INF, 10))
    add(_p("bumper_rear", "trim", "plastic", 3.0, (-1.95, 0, 0.55), (0.15, 1.70, 0.34), 0.5, (0.6, 0.3, 0.3), 5, 8))
    add(_p("rear_structure", "structure", "mild_steel", 1.2, (-1.55, 0, 0.45), (0.90, 1.30, 0.30), 0.6, (1.2, 0.8, 1.0), INF, 28,
           ("mild_steel", "high_strength_steel")))
    for name, x, y in (("wheel_fl", 1.25, 0.77), ("wheel_fr", 1.25, -0.77), ("wheel_rl", -1.25, 0.77), ("wheel_rr", -1.25, -0.77)):
        add(_p(name, "wheel", "mild_steel", 3.0, (x, y, 0.31), (0.64, 0.22, 0.62), 0.3, (0.5, 0.8, 0.5), 100, 20))
    return p


PART_TEMPLATE = _template()
PART_NAMES = [p.name for p in PART_TEMPLATE]
N_PARTS = len(PART_TEMPLATE)
PART_INDEX = {n: i for i, n in enumerate(PART_NAMES)}


def _edges():
    e = []

    def add(a, b, w):
        e.append((PART_INDEX[a], PART_INDEX[b], w))

    def both(a, b, w):
        # a and b contain "{s}" for the side letter
        for s in ("l", "r"):
            add(a.format(s=s), b.format(s=s), w)

    add("bumper_front", "beam_front", 0.8)
    add("bumper_front", "radiator", 0.4)
    both("beam_front", "rail_front_{s}", 0.9)
    add("beam_front", "radiator", 0.5)
    both("rail_front_{s}", "firewall", 0.7)
    both("rail_front_{s}", "engine", 0.5)
    add("engine", "firewall", 0.5)
    add("bonnet", "radiator", 0.3)
    add("bonnet", "windscreen", 0.4)
    add("bonnet", "firewall", 0.4)
    for s in ("l", "r"):
        add("bonnet", f"fender_{s}", 0.6)
        add(f"fender_{s}", f"wheel_f{s}", 0.4)
        add(f"fender_{s}", f"rail_front_{s}", 0.4)
        add(f"fender_{s}", f"pillar_a_{s}", 0.5)
        add(f"wheel_f{s}", f"rail_front_{s}", 0.5)
        add("windscreen", f"pillar_a_{s}", 0.7)
        add(f"pillar_a_{s}", "roof", 0.7)
        add(f"pillar_a_{s}", f"door_{s}", 0.6)
        add(f"pillar_a_{s}", "firewall", 0.5)
        add(f"pillar_a_{s}", f"sill_{s}", 0.6)
        add(f"door_{s}", f"pillar_b_{s}", 0.7)
        add(f"door_{s}", f"sill_{s}", 0.6)
        add(f"pillar_b_{s}", "roof", 0.7)
        add(f"pillar_b_{s}", f"sill_{s}", 0.7)
        add(f"pillar_b_{s}", "floor_cabin", 0.5)
        add(f"sill_{s}", "floor_cabin", 0.7)
        add(f"sill_{s}", "rear_structure", 0.4)
        add("rear_structure", f"wheel_r{s}", 0.4)
    add("windscreen", "roof", 0.6)
    add("floor_cabin", "rear_structure", 0.6)
    add("floor_cabin", "firewall", 0.6)
    add("roof", "rear_glass", 0.6)
    add("rear_glass", "rear_structure", 0.5)
    add("bumper_rear", "rear_structure", 0.8)
    return e


EDGES = _edges()


def adjacency_matrix():
    """Symmetric N x N matrix of the connection weights (0 = not connected)."""
    a = np.zeros((N_PARTS, N_PARTS), dtype=np.float32)
    for i, j, w in EDGES:
        a[i, j] = a[j, i] = w
    return a


@dataclass
class Vehicle:
    parts: list
    mass_kg: float = 1100.0

    @property
    def yaw_inertia(self):
        return self.mass_kg * (CAR_LENGTH ** 2 + CAR_WIDTH ** 2) / 12.0


def default_vehicle(mass_kg=1100.0):
    return Vehicle(list(PART_TEMPLATE), mass_kg)


def sample_vehicle(rng):
    """A random car variant: other steel grades, thickness changes, different mass."""
    scale = rng.uniform(0.9, 1.2)
    parts = []
    for p in PART_TEMPLATE:
        material = p.material
        if len(p.options) > 1 and rng.random() < 0.4:
            material = str(rng.choice(p.options))
        t = p.thickness_mm * scale * rng.uniform(0.95, 1.05)
        parts.append(replace(p, material=material, thickness_mm=round(float(t), 3)))
    return Vehicle(parts, float(rng.uniform(900, 1900)))


def part_static_features(p):
    """Numbers describing one part (material, thickness, size, position, kind...)."""
    kind = [1.0 if p.kind == k else 0.0 for k in KINDS]
    attach = np.log10(p.attach_kn) if np.isfinite(p.attach_kn) else 3.0
    return (material_features(p.material) + [p.thickness_mm, np.log10(p.mass_kg)]
            + list(p.center) + list(p.size) + [p.crush_eff, attach] + list(p.section) + kind)


PART_FEATURE_NAMES = (
    ["log_density", "log_youngs", "log_strength", "log_fracture_strain", "brittle", "thickness_mm", "log_mass"]
    + ["cx", "cy", "cz", "lx", "ly", "lz", "crush_eff", "log_attach_kn", "px", "py", "pz"]
    + [f"kind_{k}" for k in KINDS]
)


def ductility_factor(material):
    """Brittle parts (glass) break at small distortions; ductile steel tolerates a lot."""
    m = MATERIALS[material]
    return float(np.clip(m.fracture_strain / 0.15, 0.2, 1.0))
