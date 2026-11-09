"""Material table used by the vehicle parts and the obstacles.

The numbers are rough, textbook-range values (order of magnitude). They are not
taken from one specific datasheet. Replace them with values for the material you
actually want to model.

strength_mpa means the yield strength for metals and plastics, and the crushing
(plateau) strength for concrete, wood and foam.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Material:
    name: str
    density: float          # kg/m^3
    youngs_gpa: float       # GPa
    strength_mpa: float     # MPa
    fracture_strain: float  # strain at failure (dimensionless)
    brittle: bool


MATERIALS = {
    # vehicle materials
    "mild_steel": Material("mild_steel", 7850, 210, 200, 0.30, False),
    "high_strength_steel": Material("high_strength_steel", 7850, 210, 420, 0.16, False),
    "ultra_high_strength_steel": Material("ultra_high_strength_steel", 7850, 210, 950, 0.07, False),
    "aluminium": Material("aluminium", 2700, 70, 250, 0.12, False),
    "plastic": Material("plastic", 1000, 2.0, 35, 0.10, False),
    "glass": Material("glass", 2500, 70, 50, 0.001, True),
    # obstacle materials
    "concrete": Material("concrete", 2400, 30, 30, 0.003, True),
    "steel": Material("steel", 7850, 210, 250, 0.25, False),
    "wood": Material("wood", 600, 10, 40, 0.02, True),
    "foam": Material("foam", 80, 0.01, 0.15, 0.70, False),
}

MATERIAL_NAMES = sorted(MATERIALS)

# The model is meant to have a simple, readable material description.
# These four numbers go to the learning models (log scale where it helps).
MATERIAL_FEATURE_NAMES = ["log_density", "log_youngs", "log_strength", "log_fracture_strain", "brittle"]


def material_features(name):
    import math
    m = MATERIALS[name]
    return [
        math.log10(m.density),
        math.log10(m.youngs_gpa),
        math.log10(m.strength_mpa),
        math.log10(m.fracture_strain),
        1.0 if m.brittle else 0.0,
    ]
