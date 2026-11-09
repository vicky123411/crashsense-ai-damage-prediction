# Method notes

These notes describe exactly what the code does, including every constant that is a modelling
choice. Read them before trusting any number the project produces.

## 1. What this is, and what it is not

The project predicts **how a car and an obstacle look after a collision**: which car parts are
scratched, dented, crushed or torn off, how much the obstacle is deformed, and which effects
(sparks, glass, smoke, airbags, debris) make sense. The result is meant to drive visual effects in a
game engine.

It has two layers:

1. A **physics layer** (`crashdamage/physics.py`): a small, readable energy model. It is fast
   (about 0.5 ms per crash) and every step can be inspected.
2. **Learned models** (`crashdamage/models.py`): a graph neural network and some baselines that learn
   to give the same answer as the physics layer, or as an external solver if you supply its results.

It is **not** a validated crash-engineering tool. The material numbers are rough textbook values, the
car is a set of 28 boxes, and the constants in section 9 were chosen so that results land in a
plausible range (for example about 0.6 m of front crush for a 56 km/h full-width hit on a wall).
They were not fitted to measurements. Unity's built-in physics (PhysX) is also not an
engineering-grade structural solver, so the Unity side only *shows* a predicted result; it does not
compute one.

## 2. A scenario

| input | meaning |
|---|---|
| `speed_kmh` or `speed_ms` | closing speed |
| `face` | side of the car that is hit: `front`, `rear`, `left`, `right` |
| `offset` | −1…1, where along that side the obstacle touches (0 = middle). For front and rear hits positive means toward the car's left. For side hits positive means toward the front. |
| `angle_deg` | approach angle measured from the face normal (only the normal part of the speed is used) |
| `vehicle` | `mass_kg`, `thickness_scale`, per-part `materials` and `thickness_mm` |
| `obstacle` | a preset name plus overrides: `material`, `shape`, `form`, `width`, `depth`, `height`, `wall_mm`, `perimeter`, `mass_kg`, `z_base`, `after`, `stroke`, `fixed_in_place` |

In JSON, `0` and `""` mean "keep the default". Some game-engine serialisers cannot leave a field
out, so this keeps their requests simple. Per-part overrides can be a dictionary
(`{"door_l": "aluminium"}`) or a list (`[{"part": "door_l", "material": "aluminium"}]`).

```json
{
  "speed_kmh": 50, "face": "left", "offset": 0.0, "angle_deg": 0,
  "vehicle": {"mass_kg": 1300,
              "materials": [{"part": "door_l", "material": "mild_steel"}],
              "thickness_mm": [{"part": "pillar_b_l", "mm": 2.0}]},
  "obstacle": {"name": "steel_pole", "width": 0.25}
}
```

## 3. The car

Body frame: x forward, y left, z up, origin on the ground below the middle of the car.
Size 4.05 × 1.75 × 1.45 m. Every part is an axis-aligned box with a material, a thickness, a
"load-bearing perimeter" for each axis, a `crush_eff` (the share of its length that can be crushed
before it is solid), an attachment strength and a mass.

| part | kind | default material | thickness (mm) | other materials | tears off at (kN) |
|---|---|---|---:|---|---:|
| bumper_front | trim | plastic | 3 | – | 5 |
| beam_front | structure | high_strength_steel | 1.8 | aluminium, ultra_high_strength_steel | welded |
| rail_front_l / _r | structure | high_strength_steel | 1.6 | mild_steel | welded |
| radiator | mechanical | aluminium | 0.4 | – | 8 |
| engine | mechanical | aluminium | 20 | – | 60 |
| bonnet | panel | mild_steel | 0.8 | aluminium | 15 |
| fender_l / _r | panel | mild_steel | 0.7 | aluminium | 45 |
| windscreen | glass | glass | 4 | – | welded |
| pillar_a_l / _r | structure | ultra_high_strength_steel | 1.6 | high_strength_steel | welded |
| firewall | structure | high_strength_steel | 1 | – | welded |
| door_l / _r | panel | high_strength_steel | 1 | mild_steel | 25 |
| pillar_b_l / _r | structure | ultra_high_strength_steel | 1.5 | high_strength_steel | welded |
| sill_l / _r | structure | high_strength_steel | 1.4 | mild_steel | welded |
| floor_cabin | structure | high_strength_steel | 1.2 | – | welded |
| roof | panel | mild_steel | 0.8 | high_strength_steel, aluminium | welded |
| rear_glass | glass | glass | 3.5 | – | welded |
| bumper_rear | trim | plastic | 3 | – | 5 |
| rear_structure | structure | mild_steel | 1.2 | high_strength_steel | welded |
| wheel_fl / fr / rl / rr | wheel | mild_steel | 3 | – | 100 |

The parts are connected in a graph of 55 weighted edges (load paths and fixings, weights 0.3 to
0.9, see `vehicle.py`). The physics layer uses it to spread damage, and the graph network uses the
same edges for message passing.

Materials (rough values, `materials.py`):

| material | density kg/m³ | E GPa | strength MPa | fracture strain | brittle |
|---|---:|---:|---:|---:|---|
| mild_steel | 7850 | 210 | 200 | 0.3 | no |
| high_strength_steel | 7850 | 210 | 420 | 0.16 | no |
| ultra_high_strength_steel | 7850 | 210 | 950 | 0.07 | no |
| aluminium | 2700 | 70 | 250 | 0.12 | no |
| plastic | 1000 | 2 | 35 | 0.1 | no |
| glass | 2500 | 70 | 50 | 0.001 | yes |
| concrete | 2400 | 30 | 30 | 0.003 | yes |
| steel | 7850 | 210 | 250 | 0.25 | no |
| wood | 600 | 10 | 40 | 0.02 | yes |
| foam | 80 | 0.01 | 0.15 | 0.7 | no |

"Strength" is the yield strength for metals and plastic and the crushing (plateau) strength for
concrete, wood and foam.

## 4. The obstacle

An obstacle has a **material**, a **shape** (`flat`, `round`, `sharp`), a **form** (how it fails),
dimensions, a mass (or none = fixed to the ground), and what happens once its deformation stroke is
used up (`after`: `rigid` or `break`).

Plateau force (the force at which the obstacle starts to give way), with σ the material strength:

| form | force |
|---|---|
| `solid` | 0.25 · σ · width · depth |
| `thin_wall` | 0.25 · σ · wall thickness · perimeter |
| `slender` | 0.02 · σ · width · depth (fails by bending, far below crushing strength) |
| `foam` | σ · width · min(height, contact height) |

The shape acts in two ways. It changes the contact width through the preset dimensions (a sharp
edge is a few centimetres wide, so it loads few car parts but loads them hard), and it scales the
crush force of sheet-like car parts (thinner than 2.5 mm) by 1.0 / 0.9 / 0.7 for flat / round /
sharp, because sharper objects tear sheet material open at a lower force than it takes to fold it.

| preset | material | shape | form | width m | depth m | height m | mass kg | after plateau |
|---|---|---|---|---|---|---|---|---|
| concrete_wall | concrete | flat | solid | 3.0–6.0 | 0.25–0.4 | 1.5–2.5 | fixed | rigid |
| concrete_block | concrete | flat | solid | 0.5–1.2 | 0.4–0.8 | 0.5–0.9 | 500–2500 | rigid |
| steel_pole | steel | round | thin_wall | 0.15–0.45 | = width | 2.5–3.5 | fixed | rigid |
| steel_edge | steel | sharp | thin_wall | 0.04–0.12 | 0.05–0.1 | 0.8–1.2 | fixed | rigid |
| wood_fence | wood | flat | slender | 1.0–3.0 | 0.02–0.05 | 1.0–1.8 | fixed | break |
| wood_post | wood | round | slender | 0.12–0.3 | = width | 1.5–2.5 | fixed | break |
| foam_barrier | foam | flat | foam | 1.0–2.5 | 0.4–1.0 | 0.6–1.2 | fixed | rigid |
| steel_crate | steel | flat | thin_wall | 0.5–1.2 | 0.4–0.9 | 0.4–0.9 | 80–600 | rigid |

Ranges are sampled uniformly when a dataset is generated; a single scenario uses the middle values
unless you override them.

## 5. The physics layer, step by step

**5.1 Energy.** The two bodies meet and end up (nearly) moving together, so the energy that has to
go into deformation is

    1/m_car_eff = 1/M + arm² / I_z          (I_z = M (L² + W²) / 12, arm = distance of the hit from the centre line)
    μ           = 1 / (1/m_car_eff + 1/m_obstacle)      (no obstacle term if it is fixed)
    v_n         = v · cos(angle)
    E_diss      = ½ μ v_n² (1 − e²),   e = 0.05

The velocity change of the car is Δv = (1 + e) μ v_n / M. A free obstacle leaves with
(1 + e) μ v_n / m_obstacle.

**5.2 Contact patch.** The obstacle touches a strip of the chosen face: its width (capped at the face
width) centred at `offset · (half face − width / 2)`, and the height range
[`z_base`, `z_base + height`] clipped to the car. A part's **overlap** is the share of its section
covered by the patch (along the face times vertically). Parts with overlap above 0.02 are "in the line
of fire". Each also has a `depth_start` (distance from the face to the part's near side) and a
`stroke` (its length along the push direction times `crush_eff`).

**5.3 Crush force.** A part in the patch resists with

    F_i = 0.25 · σ_i · t_i · s_i · overlap_i · cut          (s_i = load-bearing perimeter along the push axis)

over its stroke, starting at its `depth_start`. After the stroke it is used up and no longer resists,
except solid blocks (`crush_eff < 0.15`: the engine and the cabin floor), which pack and then resist
with min(8 · F_i, 1.5 MN) until the far side of the car. The energy a part can take is its
**capacity** C_i = F_i · stroke_i. Parts at the same depth add their forces; parts behind each other
come one after the other.

**5.4 Solver.** The car's total resistance is a staircase over the depth, the obstacle's is its plateau
force up to its stroke and then rigid (1e9 N) or zero (`break`). The two bodies push on each other
with the same force, so the one with the lower current resistance gives way. The code marches the
bodies into each other in steps where nothing changes, subtracting force × distance from E_diss until
it is used up. This gives the damage of **both** bodies. Three special outcomes are reported:

* the obstacle breaks (`broken_through`): the remaining energy stays as motion, the car keeps going;
* nothing is left to resist in the line of the obstacle (`went_through_car`): a narrow hard object
  went through the car's structure. The depth is capped at the size of the car;
* otherwise the energy is fully absorbed. In every case
  `energy_car + energy_obstacle + energy_unabsorbed = energy_total`, which the tests check.

**5.5 Damage score.** For a loaded part, d = E_part / C_i, clipped to 0…1, where E_part is the energy it
actually absorbed. A loaded part that did not yield still gets 0.1 · min(1, F_max / F_i) for light
marks.

**5.6 Spreading through the graph.** Force travels along the edges for two hops (the second hop is
weakened by 0.6). A part cannot pass on more than it can carry,
F_yield = 0.25 · σ · t · mean(s):

    carried_i = min(max(F_direct_i, F_in_i), F_yield_i)
    F_in_j    = decay · max_i( A_ij · carried_i )
    d_in_j    = min( 0.8 · min(1, F_in_j / F_yield_j) / g_j ,  decay · max_i( A_ij · d_i ) )
    d_j       = max(d_direct_j, d_in_j)

`g` is a ductility factor, clip(fracture strain / 0.15, 0.2, 1): glass (g = 0.2) is damaged five
times more easily than steel by the same force. The second term stops a part from being worse off
than the neighbour it hangs on. Because both pathways grow with speed, the damage of every part is
non-decreasing in speed (also a test).

**5.7 Derived outputs.**

* Level from damage with edges 0.05 / 0.2 / 0.5 / 0.85: intact, scratched, dented, crushed, destroyed.
* A part **detaches** when the force at the part ≥ its attachment strength and its damage ≥ 0.3.
* **Cabin intrusion** = max(0, crush depth − cabin depth), with cabin depth 0.90 / 0.80 / 0.36 / 0.36 m for
  front / rear / left / right.
* **Airbags**: front if the front is hit within 30° and Δv ≥ 5.5 m/s; side if a side is hit and Δv ≥ 4.0 m/s
  (assumed trigger levels).
* **Obstacle damage** = energy it took / (plateau force × stroke), set to 1 if it breaks.
* **Effect hints** are simple rules: sparks (hard obstacle, metal hit, > 5 kJ), glass shards (a glass part
  crushed), smoke and fluid leak (radiator or engine crushed), debris (something torn off or destroyed),
  dust (concrete or wood obstacle damaged).

## 6. Damage report (the JSON the game engine gets)

```
parts[]     id, damage (0..1), level, level_name, detached, direct
obstacle    name, material, damage, level, level_name, deformation_m, broken_through, speed_after_ms
summary     crush_depth_m, cabin_intrusion_m, peak_force_kn, energy_total_kj, energy_car_kj,
            energy_obstacle_kj, energy_unabsorbed_kj, went_through_car, delta_v_ms
effects     sparks, glass_shards, smoke, fluid_leak, debris, dust, airbag_front, airbag_side
impact      face, offset, angle_deg, speed_ms, speed_kmh
engine      "physics" or "model"
```

When a trained model answers, the quantities it does not predict (`energy_car_kj`,
`energy_obstacle_kj`, `energy_unabsorbed_kj`, `deformation_m`, `broken_through`) are left out, and
`direct` means "inside the contact patch".

## 7. Learned models

**Inputs.** For every part 29 numbers: material (log density, log stiffness, log strength, log
fracture strain, brittle flag), thickness, log mass, position, size, `crush_eff`, log attachment
strength, load-bearing perimeters, part kind (one-hot), and five contact numbers (overlap, depth start, stroke,
distance to the contact point, in-patch flag). For the scenario 36 numbers: face, speed, normal speed,
angle, offset, patch width, centre and height range, lever arm, log car mass, and 22 obstacle
numbers (material, shape, form, size, mass, ...). Nothing that comes out of the solver is an input.

Optional **physics features** (`--physics-features`): log impact energy, log total capacity and their
ratio for the scenario, and log capacity and energy-to-capacity ratio for each part. They are closed-form and
need no solver run.

**Outputs.** Per part: damage and a "falls off" logit. Per scenario: crush depth, cabin intrusion,
peak force, obstacle damage, and three flags (front airbag, side airbag, went through the car).
Levels are derived from the predicted damage with the same thresholds as above.

**Models** (`models.py`, `baselines.py`):

| name | what it is |
|---|---|
| `naive_energy` | no learning: impact energy shared over the loaded parts in proportion to their capacity |
| `gbm` | scikit-learn histogram gradient boosting. Per-part targets use one row per (scenario, part) |
| `mlp` | one MLP on the whole flattened input, no graph |
| `node_mlp` | the graph network with zero message-passing rounds: each part on its own |
| `gnn` | the graph network: 3 rounds of message passing on the part graph, `m_i = Σ_j Â_ij W h_j`, `h_i ← LayerNorm(h_i + MLP[h_i, m_i, g])`, hidden size 96, readout by mean and max pooling |
| `gnn_phys` | `gnn` with the physics features as extra inputs |
| `gnn_phys_energy` | `gnn_phys` plus an energy-consistency term in the loss |

The graph is small and fixed (28 nodes), so message passing is a matrix product with the row-normalised
weighted adjacency. No graph library is needed.

**Loss.** 5 · MSE(damage) + BCE(detach, positive class weighted 4) + smooth-L1(standardised regression
targets) + BCE(flags). The optional **energy-consistency** term penalises predictions that give loaded
parts more energy than exists:

    relu( Σ_i d̂_i · C_i − E_diss − 0.1 · Σ_i C_i ) / E_diss

(the 0.1 allowance matches the light marks the labels give to unyielded parts). AdamW, learning rate
2e-3 with cosine decay, batch size 128, early stopping on the validation loss.

## 8. Evaluation

Data come from `dataset.generate`: random cars (mass 900–1900 kg, thickness scale 0.9–1.2, 40 % chance
per part to switch to another allowed material), a random obstacle preset with random dimensions,
face probabilities 0.45 / 0.10 / 0.225 / 0.225 (front / rear / left / right), offset uniform in −1…1,
angle N(0, 12°) clipped to ±40°, speed uniform 4–36 m/s.

Splits (`dataset.make_split`):

* `random`: ordinary 70 / 15 / 15 split.
* `speed`: train on crashes below 24 m/s, test on faster ones (extrapolation in speed).
* `obstacle`: `steel_crate` never appears in training.
* `face`: the rear is never hit in training.

Metrics (`metrics.py`): damage MAE, RMSE and R² over all parts (and for loaded and unloaded parts
separately); level accuracy, accuracy within one level and macro-F1; detach precision, recall and F1;
MAE of crush depth, intrusion, peak force and obstacle damage; accuracy and F1 of the three flags.

**How to read the results.** The labels come from the physics layer, so a good score means *the model
reproduces that physics layer*, including on cases it has not seen. It says nothing about agreement
with real crashes. The benefit shown is speed (a forward pass for many scenarios at once, and a
model that can be retrained on external labels) and generalisation behaviour, not physical truth.

## 9. Constants and choices

| constant | value | where | role |
|---|---|---|---|
| `FOLD` | 0.25 | obstacles.py | share of σ·t·perimeter that a folding thin wall resists with |
| `SOLID_K` | 0.25 | obstacles.py | same for a solid block |
| `SLENDER_K` | 0.02 | obstacles.py | slender objects fail by bending |
| `RESTITUTION` | 0.05 | physics.py | nearly plastic impact |
| `DENS`, `F_BLOCK_CAP` | 8, 1.5 MN | physics.py | force of a packed solid block |
| `BLOCK_EFF` | 0.15 | physics.py | parts that can crush less than this act as solid blocks |
| `MIN_OVERLAP` | 0.02 | physics.py | smaller overlaps are ignored |
| `ELASTIC_FLOOR` | 0.1 | physics.py | light marks on loaded parts that did not yield |
| `PROPAGATION_HOPS`, `HOP_DECAY` | 2, 0.6 | physics.py | spreading through the graph |
| `INDIRECT_GAIN` | 0.8 | physics.py | damage per unit of force ratio for parts hit only indirectly |
| `DETACH_MIN_DAMAGE` | 0.3 | physics.py | a part must be damaged to tear off |
| `LEVEL_EDGES` | 0.05 / 0.2 / 0.5 / 0.85 | physics.py | damage to level |
| `SHAPE_CUT`, `SHEET_MM` | 1.0 / 0.9 / 0.7, 2.5 mm | physics.py | effect of the obstacle shape on sheet parts |
| `AIRBAG_FRONT_DV`, `AIRBAG_SIDE_DV` | 5.5, 4.0 m/s | physics.py | assumed airbag triggers |
| `CABIN_DEPTH` | 0.90 / 0.80 / 0.36 / 0.36 m | vehicle.py | where the passenger cell starts |

## 10. Training on results from a finite-element solver

`crashdamage/external.py` reads two CSV files and builds the same arrays as the generator, so every
model in the project can be trained on solver output instead (`python -m crashdamage import-fe
--cases cases.csv --parts parts.csv`). The column layout is at the top of that file. You decide how a solver result
becomes a 0…1 damage number per part, for example peak plastic strain divided by failure strain, or
the crushed share of the part's length, and you should write that definition next to the dataset.
`write_csv` writes the same layout from physics-layer results (used by the tests).

The reader is tested on files written by this project. It has not been tried on real OpenRadioss or
LS-DYNA output, so check the first few cases by hand. The car model also stays this 28-part car; a
different vehicle needs its own part list.

## 11. Known limits

* One car layout (a compact car made of boxes). Variants differ in mass, materials and thickness only.
* Quasi-static energy method: no time history, no rebound beyond e = 0.05, no friction, no rotation
  beyond the effective-mass term, no sliding off a glancing hit.
* Damage scores are shares of what a part can absorb, not strains or stresses.
* Damage spreads over the part graph with hand-set connection weights. Parts on the far side of a hit can come
  out more damaged than in a real test (a left-side pole hit marks the right sill as crushed), and a stronger
  rail passes more force to its neighbours, which can tear them off even though the crush is shallower.
* Peak force is the largest value of a stepwise force curve. It jumps when the car reaches a stiff block such
  as the engine, so treat it as a rough indicator.
* Narrow hard obstacles (the steel edge and the steel pole) often pass through the whole car in the model. The
  report then sets `went_through_car` and gives the unabsorbed energy.
* The level thresholds, airbag triggers and effect rules are assumptions for visual purposes.
* No occupants, no fire, no fracture modelling inside parts.
* Nothing here has been compared with crash-test data.
