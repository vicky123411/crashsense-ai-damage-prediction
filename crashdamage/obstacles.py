"""Things the car can hit.

An obstacle has a material, a shape (flat / round / sharp), a structural form
(how it fails), its size, and either a mass (it can be pushed) or no mass (fixed
to the ground). Presets below cover the usual cases; every number can be
overridden in a scenario file.
"""
from dataclasses import dataclass, replace

from .materials import MATERIALS

FOLD = 0.25        # share of (strength x thickness x perimeter) that a folding thin wall resists with
SOLID_K = 0.25     # same idea for a solid block loaded across its section
SLENDER_K = 0.02   # slender things (fence boards, posts) fail by bending, far below crushing strength

SHAPES = ["flat", "round", "sharp"]
FORMS = ["solid", "thin_wall", "slender", "foam"]
AFTER = ["rigid", "break"]


@dataclass(frozen=True)
class Obstacle:
    name: str
    material: str
    shape: str
    form: str
    width: float            # m, across the car
    depth: float            # m, section depth along the impact direction
    height: float           # m
    wall_mm: float = 0.0    # wall thickness for thin-walled forms
    perimeter: float = 0.0  # m, loaded perimeter for thin-walled forms
    mass_kg: float = None   # None = fixed to the ground
    after: str = "rigid"    # what happens once the plateau stroke is used up
    stroke: float = 0.02    # m of deformation at the plateau force
    z_base: float = 0.0     # m, height of the lowest point

    @property
    def fixed(self):
        return self.mass_kg is None

    def plateau_force(self, contact_width, contact_height=0.6):
        """Force (N) at which the obstacle starts to give way."""
        sigma = MATERIALS[self.material].strength_mpa * 1e6
        w = min(self.width, contact_width)
        if self.form == "solid":
            return SOLID_K * sigma * w * self.depth
        if self.form == "thin_wall":
            return FOLD * sigma * self.wall_mm * 1e-3 * self.perimeter
        if self.form == "slender":
            return SLENDER_K * sigma * w * self.depth
        if self.form == "foam":
            return sigma * w * min(self.height, contact_height)
        raise ValueError(f"unknown form {self.form}")


# (low, high) ranges are sampled uniformly; plain numbers are fixed
PRESETS = {
    "concrete_wall": dict(material="concrete", shape="flat", form="solid", width=(3.0, 6.0), depth=(0.25, 0.4),
                          height=(1.5, 2.5), mass_kg=None, after="rigid", stroke=0.01),
    "concrete_block": dict(material="concrete", shape="flat", form="solid", width=(0.5, 1.2), depth=(0.4, 0.8),
                           height=(0.5, 0.9), mass_kg=(500, 2500), after="rigid", stroke=0.02),
    "steel_pole": dict(material="steel", shape="round", form="thin_wall", width=(0.15, 0.45), depth="w",
                       height=(2.5, 3.5), wall_mm=(4, 10), perimeter="2w", mass_kg=None, after="rigid", stroke=0.08),
    "steel_edge": dict(material="steel", shape="sharp", form="thin_wall", width=(0.04, 0.12), depth=(0.05, 0.1),
                       height=(0.8, 1.2), wall_mm=(10, 25), perimeter=(0.15, 0.3), mass_kg=None, after="rigid", stroke=0.03),
    "wood_fence": dict(material="wood", shape="flat", form="slender", width=(1.0, 3.0), depth=(0.02, 0.05),
                       height=(1.0, 1.8), mass_kg=None, after="break", stroke=0.05),
    "wood_post": dict(material="wood", shape="round", form="slender", width=(0.12, 0.30), depth="w",
                      height=(1.5, 2.5), mass_kg=None, after="break", stroke=0.04),
    "foam_barrier": dict(material="foam", shape="flat", form="foam", width=(1.0, 2.5), depth=(0.4, 1.0),
                         height=(0.6, 1.2), mass_kg=None, after="rigid", stroke="0.6d"),
    "steel_crate": dict(material="steel", shape="flat", form="thin_wall", width=(0.5, 1.2), depth=(0.4, 0.9),
                        height=(0.4, 0.9), wall_mm=(1.5, 4), perimeter="w+d", mass_kg=(80, 600), after="rigid", stroke=0.15),
}
PRESET_NAMES = list(PRESETS)


def _draw(v, rng):
    if isinstance(v, tuple):
        return float(rng.uniform(v[0], v[1]))
    return v


def sample_obstacle(rng, name=None):
    if name is None:
        name = PRESET_NAMES[int(rng.integers(len(PRESET_NAMES)))]
    spec = PRESETS[name]
    vals = {}
    for key in ("width", "depth", "height", "wall_mm", "perimeter", "mass_kg", "stroke"):
        if key in spec and not isinstance(spec[key], str):
            vals[key] = _draw(spec[key], rng)
    if spec.get("depth") == "w":
        vals["depth"] = vals["width"]
    if spec.get("perimeter") == "2w":
        vals["perimeter"] = 2.0 * vals["width"]
    if spec.get("perimeter") == "w+d":
        vals["perimeter"] = vals["width"] + vals["depth"]
    if spec.get("stroke") == "0.6d":
        vals["stroke"] = 0.6 * vals["depth"]
    z_base = 0.0 if rng.random() < 0.7 else float(rng.uniform(0.1, 0.5))
    return Obstacle(name=name, material=spec["material"], shape=spec["shape"], form=spec["form"],
                    after=spec["after"], z_base=z_base,
                    width=vals["width"], depth=vals["depth"], height=vals["height"],
                    wall_mm=vals.get("wall_mm", 0.0), perimeter=vals.get("perimeter", 0.0),
                    mass_kg=vals.get("mass_kg"), stroke=vals["stroke"])


def default_obstacle(name):
    """Middle-of-the-range version of a preset."""

    class Mid:
        def uniform(self, lo, hi):
            return 0.5 * (lo + hi)

        def random(self):
            return 0.0

        def integers(self, n):
            return 0

    return sample_obstacle(Mid(), name)


def obstacle_from_dict(d):
    """Obstacle from JSON.

    Use a preset name (``"name": "steel_pole"``) and override what you like. A custom name needs a
    ``"base"`` preset to start from. In the JSON, 0 and "" mean "keep the preset value" because
    some game-engine serialisers cannot leave a field out; use ``"fixed": true`` (or ``"fixed_in_place"``)
    for an obstacle that cannot be pushed.
    """
    d = dict(d)
    name = d.get("name") or d.get("preset") or "concrete_wall"
    base_name = d.get("base") or name
    if base_name not in PRESETS:
        raise ValueError(f"unknown obstacle {base_name!r}; presets: {', '.join(PRESET_NAMES)}")
    over = {}
    for key, value in d.items():
        if key not in Obstacle.__dataclass_fields__ or key == "name" or value is None or value == "":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0 and key != "z_base":
            continue
        over[key] = value
    if "material" in over and over["material"] not in MATERIALS:
        raise ValueError(f"unknown material {over['material']!r}")
    for key, allowed in (("shape", SHAPES), ("form", FORMS), ("after", AFTER)):
        if key in over and over[key] not in allowed:
            raise ValueError(f"{key} must be one of {allowed}")
    ob = replace(default_obstacle(base_name), name=name, **over)
    if d.get("fixed") or d.get("fixed_in_place"):
        ob = replace(ob, mass_kg=None)
    return ob


def obstacle_features(ob):
    """Numbers describing an obstacle (for the learning models)."""
    import math
    from .materials import material_features
    shape = [1.0 if ob.shape == s else 0.0 for s in SHAPES]
    form = [1.0 if ob.form == f else 0.0 for f in FORMS]
    mass = math.log10(ob.mass_kg) if ob.mass_kg else 0.0
    return (material_features(ob.material) + shape + form +
            [ob.width, ob.depth, ob.height, ob.wall_mm, ob.perimeter, mass, 1.0 if ob.fixed else 0.0,
             1.0 if ob.after == "break" else 0.0, ob.stroke, ob.z_base])


OBSTACLE_FEATURE_NAMES = (
    ["ob_log_density", "ob_log_youngs", "ob_log_strength", "ob_log_fracture_strain", "ob_brittle"]
    + [f"shape_{s}" for s in SHAPES] + [f"form_{f}" for f in FORMS]
    + ["ob_width", "ob_depth", "ob_height", "ob_wall_mm", "ob_perimeter", "ob_log_mass", "ob_fixed", "ob_breaks",
       "ob_stroke", "ob_z_base"]
)
