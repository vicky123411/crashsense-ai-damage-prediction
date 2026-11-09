# CrashSense AI: damage prediction for vehicle–obstacle collisions

Predicts how a car **and the object it hits** look after a crash, part by part, so that a game engine can show it.

You give it a speed, where the car is hit (side, position along that side, angle), what the car is made of
(steel grade and sheet thickness per part, total mass) and what it hits (concrete, steel, wood or foam; flat, round
or sharp; fixed or free to move). You get back:

* a damage value and a level (intact, scratched, dented, crushed, destroyed) for each of 28 car parts,
* which parts tear off,
* the damage of the obstacle, and whether it broke or was pushed away,
* crush depth, cabin intrusion, peak force and speed change,
* effect flags for the game: sparks, glass shards, smoke, fluid leak, debris, dust, front and side airbag.

The Unity scripts in `unity/` turn that report into dents, cracked glass, loose parts and particles.

The project grew out of a Unity car-crash simulation (Ramp Run) and a simple question: can the look of a crash be
predicted from speed, materials and hit position, so that artists and tools can set up post-crash effects faster?

> **Please read first.** The labels used here do not come from crash tests or finite-element simulations. They come
> from a small physics layer in this repository (an energy model of 28 boxes). The neural networks are trained to
> reproduce that layer, so the scores below measure *agreement with the physics layer on synthetic data*, not accuracy
> against real crashes. The physics layer is a plausibility model for visuals: its constants were chosen to give
> believable magnitudes and were not fitted to measurements. Unity's physics engine (PhysX) is not a structural
> solver either, so the Unity side only displays a predicted result. What the repository offers is a working pipeline
> that can later be trained on real solver output (there is an importer for that), not engineering accuracy, and it
> must not be used to judge vehicle safety. All assumptions and constants are in [`docs/method.md`](docs/method.md).

## How it fits together

```
 scenario (JSON) ──► physics layer ──► labelled data (damage per part, obstacle damage, flags)
                          │                      │
                          │                      ▼
                          │              graph neural network  (also MLP and boosting baselines)
                          ▼                      │
                    damage report  ◄─────────────┘            (same JSON from either engine)
                          │
                    HTTP server on localhost ──► Unity: dents, glass, loose parts, particles
```

* **Physics layer** (`crashdamage/physics.py`): impact energy from speed and masses; a contact patch from the size
  and position of the obstacle; crush forces from material, thickness and cross-section; a solver in which the weaker
  body (car or obstacle) gives way first; and damage that spreads along the load paths of the car
  (door → pillar → windscreen, ...). Damage is the share of a part's crush capacity that was used.
* **Learned models** (`crashdamage/models.py`): a graph neural network on the part graph, compared with a flat MLP,
  a per-part MLP, gradient boosting and a no-learning energy baseline.
* **Unity bridge** (`unity/`, `crashdamage/serve.py`): a small HTTP server and C# scripts.

## Quick start

```bash
pip install -r requirements.txt          # numpy, scikit-learn, torch (CPU is enough), pytest, matplotlib

python -m crashdamage simulate examples/side_pole_29kmh.json    # physics layer: prints the damage report
python -m crashdamage generate --n 8000                         # labelled dataset -> data/synthetic.npz
python -m crashdamage train --model gnn --physics-features      # trains, prints test scores, saves models/gnn.pt
python -m crashdamage benchmark                                 # every model on every split (about 45 minutes on 2 CPU cores)
python -m crashdamage serve --model models/gnn_demo.pt          # server for Unity on 127.0.0.1:8765
pytest                                                          # 47 tests
```

`pip install -e .` also gives you a `crashdamage` command. Tested with Python 3.13 (numpy 2.5, scikit-learn 1.9,
PyTorch 2.15, CPU only). A small demo model, `models/gnn_demo.pt`, is included.
It was trained with `python -m crashdamage train --model gnn --physics-features --epochs 60 --out models/gnn_demo.pt`
on the random split of the 8000 synthetic scenarios (5600 for training, 1200 for validation; it stopped early at
epoch 48). On the 1200 held-out test scenarios its damage error is 0.0325 on average, the damage level is right in
86% of the cases (99% within one level), detaching parts is predicted with F1 0.95, and the average errors are
0.065 m for crush depth and 0.047 m for cabin intrusion.

## Example

`examples/side_pole_29kmh.json`: the left side of a 1300 kg car meets a 25 cm steel pole at 29 km/h, in the middle
of the door.

```json
{"speed_kmh": 29, "face": "left", "offset": 0.0, "angle_deg": 0,
 "vehicle": {"mass_kg": 1300}, "obstacle": {"name": "steel_pole", "width": 0.25}}
```

Physics layer: crush depth 0.64 m, cabin intrusion 0.28 m, peak force 115 kN, 42 kJ absorbed by the car and none by
the pole (it stays intact), speed change 8.5 m/s. Effects: sparks, debris, side airbag. The most damaged parts:

| part | damage | level | torn off |
|---|---:|---|---|
| door_l | 1.00 | destroyed | yes |
| sill_l | 1.00 | destroyed | no |
| floor_cabin | 1.00 | destroyed | no |
| sill_r | 0.70 | crushed | no |
| roof | 0.67 | crushed | no |
| rear_structure | 0.60 | crushed | no |
| pillar_b_l | 0.51 | crushed | no |
| pillar_b_r | 0.50 | crushed | no |

The demo model answers the same scenario in the same range
(`python -m crashdamage simulate examples/side_pole_29kmh.json --model models/gnn_demo.pt --engine model`):

| | physics layer | demo model |
|---|---:|---:|
| crush depth (m) | 0.64 | 0.71 |
| cabin intrusion (m) | 0.28 | 0.29 |
| peak force (kN) | 115 | 125 |
| door_l damage | 1.00, torn off | 1.00, torn off |
| sill_r damage | 0.70 | 0.73 |
| roof damage | 0.67 | 0.55 |
| firewall damage | 0.30 | 0.42 |

The effect flags (sparks, debris, side airbag) and the intact pole agree. Over all 28 parts the damage values differ
by 0.024 on average for this scenario. Damage levels are cut from the damage value at fixed thresholds, so a model
value just below a threshold gets the lower level, as with the right B-pillar (0.50 "crushed" against just under 0.50
"dented").

The right sill and right B-pillar are on the far side of the hit. They come out "crushed" because damage spreads
over hand-set connection weights, which is probably more than a real pole test would show; see the limitations
below.

All scenarios in `examples/` through the physics layer (`python scripts/examples_table.py`):

| example | speed (km/h) | crush depth (m) | cabin intrusion (m) | peak force (kN) | parts crushed or destroyed | torn off | obstacle | effects |
|---|---:|---:|---:|---:|---:|---|---|---|
| foam_barrier_56kmh | 56 | 0.46 | 0.00 | 236 | 7 | bumper_front, radiator | crushed | smoke, fluid leak, debris, airbag front |
| free_concrete_block_40kmh | 40 | 0.64 | 0.00 | 1015 | 5 | bumper_front, radiator, engine | intact | sparks, smoke, fluid leak, debris, airbag front |
| front_wall_56kmh | 56 | 0.63 | 0.00 | 1575 | 10 | bumper_front, radiator, bonnet | intact | sparks, smoke, fluid leak, debris, airbag front |
| mild_vs_ultra_high_strength_front | 56 | 0.48 | 0.00 | 572 | 7 | bumper_front, radiator, bonnet, fender_l, fender_r | intact | sparks, smoke, fluid leak, debris, airbag front |
| offset_front_pole_50kmh | 50 | 0.76 | 0.00 | 328 | 9 | bumper_front, radiator, engine, bonnet | destroyed | sparks, smoke, fluid leak, debris, airbag front |
| side_pole_29kmh | 29 | 0.64 | 0.28 | 115 | 8 | door_l | intact | sparks, debris, airbag side |
| wood_fence_50kmh | 50 | 0.07 | 0.00 | 49 | 1 | bumper_front | destroyed (broke) | debris, dust, airbag front |

A request may also change materials and thicknesses of single parts and the obstacle (every field is optional; 0 or
an empty string keeps the preset):

```json
"vehicle": {"mass_kg": 1100, "materials": [{"part": "beam_front", "material": "ultra_high_strength_steel"}]}
```

## Results on synthetic data

8000 random crashes labelled by the physics layer: random cars (mass, steel grades, thickness), eight obstacle
presets, all four sides, random offset, angle and speed (4 to 36 m/s). Neural models were trained with 3 seeds, at
most 40 epochs, early stopping on a validation set; the table shows mean ± standard deviation on the test rows.
Four ways of splitting the data:

* **random**: ordinary 70/15/15 split
* **speed**: trained below 24 m/s (86 km/h), tested above it
* **obstacle**: the `steel_crate` obstacle never appears in training
* **face**: the rear of the car is never hit in training

**Damage MAE (mean absolute error of the 0-1 damage value, lower is better)**

| model | random | speed |
|---|---:|---:|
| naive_energy | 0.2699 | 0.3091 |
| gbm | 0.0646 | 0.0796 |
| mlp | 0.0649 ± 0.0007 | 0.1052 ± 0.0024 |
| node_mlp | 0.0548 ± 0.0006 | 0.0735 ± 0.0005 |
| gnn | 0.0353 ± 0.0004 | 0.0529 ± 0.0006 |
| gnn_phys | 0.0337 ± 0.0001 | 0.0524 ± 0.0004 |
| gnn_phys_energy | 0.0360 ± 0.0004 | 0.0542 ± 0.0015 |

**Damage level macro-F1 (five levels, higher is better)**

| model | random | speed |
|---|---:|---:|
| naive_energy | 0.275 | 0.239 |
| gbm | 0.672 | 0.631 |
| mlp | 0.682 ± 0.002 | 0.583 ± 0.006 |
| node_mlp | 0.729 ± 0.003 | 0.668 ± 0.006 |
| gnn | 0.820 ± 0.001 | 0.764 ± 0.003 |
| gnn_phys | 0.827 ± 0.001 | 0.761 ± 0.010 |
| gnn_phys_energy | 0.813 ± 0.001 | 0.757 ± 0.008 |

The variants: `naive_energy` shares the impact energy over the loaded parts by capacity and learns nothing; `gbm`
is gradient boosting; `mlp` sees the whole scenario as one flat vector; `node_mlp` is a per-part network without
message passing; `gnn` passes messages along the part graph; `gnn_phys` also gets the energy and capacity numbers
of the physics layer as inputs, and `gnn_phys_energy` adds a loss term that penalises damage totals the energy
cannot pay for. `naive_energy` and `gbm` were run once, the neural models three times.

What the first two splits show (the `obstacle` and `face` splits are still being run and will be added here):

* On the random split the graph network roughly halves the damage error of the flat MLP and of gradient boosting
  (0.035 against 0.065), and its macro-F1 over the five damage levels is 0.82 against 0.67 to 0.68.
* Asking for speeds above the training range makes every learned model worse, and the flat MLP worst of all
  (0.105). The graph network is still the best (0.053), so the part graph also helps with extrapolation.
* Giving the graph network the energy and capacity numbers of the physics layer (`gnn_phys`) helps a little on the
  random split (0.034 against 0.035) and not visibly on the speed split. The energy-consistency loss
  (`gnn_phys_energy`) did not help; it is slightly worse than `gnn_phys` on both.
* The no-learning baseline (`naive_energy`) is worse than always predicting the average damage (R² below zero), so
  the learned models do pick up more than a simple energy share.

## Repository layout

```
crashdamage/
  materials.py   material table (rough textbook values)
  vehicle.py     the 28-part car, its part graph, random car variants
  obstacles.py   obstacle presets, forms, shapes
  physics.py     energy model: contact patch, crush forces, solver, damage spreading
  features.py    the numbers that go into the models
  dataset.py     random scenarios, labelling, splits
  models.py      graph network and MLPs
  baselines.py   energy baseline and gradient boosting
  train.py       training loop, saving and loading
  metrics.py     scores
  evaluate.py    benchmark over models and splits
  export.py      damage report JSON (physics layer or model)
  serve.py       HTTP server for the game engine
  external.py    CSV import for labels from OpenRadioss / LS-DYNA runs
  cli.py         python -m crashdamage ...
unity/           C# scripts and set-up notes
examples/        scenario files
docs/method.md   everything the code assumes
scripts/         example table, result table and plot, timing
results/         benchmark output
tests/           pytest
```

## Limitations

* **No real data.** There is no comparison with crash tests or solver runs. The constants were chosen for plausible
  magnitudes (a 1.1 t car into a rigid wall at 56 km/h crushes about 0.6 m), not fitted.
* **One car layout**: a compact car made of 28 boxes. Variants differ in mass, materials and thickness only.
* **Quasi-static energy model**: no time history, only a small restitution, no friction, no sliding off a glancing hit.
* **Damage spreading is a heuristic.** It follows hand-set connection weights, so far-side parts can look more damaged
  than in a real test (the right sill in the example above), and a stronger rail can pass more force to its
  neighbours and tear the fenders off even though the crush is shallower (`mild_vs_ultra_high_strength_front`).
* **Peak force is jumpy.** It is the largest value of a stepwise force curve, and it rises sharply when the car
  reaches a stiff block such as the engine. Treat it as a rough indicator.
* **Narrow hard objects go through the car.** In 11% of the generated scenarios the obstacle passes through
  the whole car in the model (the steel edge in 53% of its scenarios, the steel pole in 17%). The report then
  says `went_through_car` and gives the unabsorbed energy; read the damage as "destroyed along the line of the hit".
* **Models inherit the physics layer's assumptions** and are only as good as that layer; the scores are agreement
  scores, not accuracy.
* **The Unity scripts were never run in Unity.** They compile (C# 9) against stand-in Unity types, which only catches
  typos and wrong names between the scripts, not problems with the real Unity API. Expect small fixes on first import.
* **The CSV importer for solver labels** was only tested on files this project writes itself.

## Ideas for next steps

* Train on real solver data (OpenRadioss or LS-DYNA runs, or a public crash dataset) through `external.py`, and
  calibrate the constants against it.
* More vehicle layouts, and car-to-car crashes where both bodies are deformable.
* A mesh-level graph network for the shape of the dent, not only per-part damage.
* Try the Unity scripts in the Ramp Run scene and tune the dent and effect parameters with artists.

## License

MIT, see [`LICENSE`](LICENSE).
