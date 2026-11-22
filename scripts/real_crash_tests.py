"""Test the physics layer on real crash tests (NHTSA NCAP full-frontal rigid barrier, ~56 km/h).

For every test in data_real/nhtsa_ncap_frontal.csv the physics layer is run on its own compact car
with the test's mass and speed, head-on into a rigid wall, and its crush depth is compared with the
maximum static crush measured on the real car.

Then one constant is calibrated: a single multiplier on every car part's crushing force (the
kappa = 0.25 of Eq. 2). It is fitted by leave-one-out: for each test the multiplier is chosen on the
other 14 tests and then used to predict the held-out one, so the reported error is not fitted to
the test it scores.

Caveats: the real cars are not the compact car of the model (they are bigger and heavier); the
tests measure residual crush after spring-back, the layer gives the deepest crush during impact;
only crush depth is compared, not damage to single parts.

    python scripts/real_crash_tests.py
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crashdamage import physics  # noqa: E402
from crashdamage.obstacles import default_obstacle  # noqa: E402
from crashdamage.physics import Scenario, simulate  # noqa: E402
from crashdamage.vehicle import default_vehicle  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_FOLD = physics.FOLD


def load_tests(path=os.path.join(HERE, "..", "data_real", "nhtsa_ncap_frontal.csv")):
    with open(path) as f:
        rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
    for r in rows:
        for k in ("speed_kmh", "test_mass_kg", "max_static_crush_mm"):
            r[k] = float(r[k])
    return rows


def crush_depth(mass, speed_kmh, scale=1.0):
    physics.FOLD = BASE_FOLD * scale
    try:
        sc = Scenario(default_vehicle(mass), default_obstacle("concrete_wall"), face="front", offset=0.0,
                      angle_deg=0.0, speed=speed_kmh / 3.6)
        return simulate(sc).crush_depth
    finally:
        physics.FOLD = BASE_FOLD


SCALES = np.round(np.arange(0.5, 4.01, 0.05), 3)


def main():
    tests = load_tests()
    meas = np.array([t["max_static_crush_mm"] / 1000 for t in tests])
    table = np.array([[crush_depth(t["test_mass_kg"], t["speed_kmh"], s) for s in SCALES] for t in tests])
    base = np.array([crush_depth(t["test_mass_kg"], t["speed_kmh"]) for t in tests])

    def best_scale(idx):
        err = np.abs(table[idx] - meas[idx, None]).mean(axis=0)
        return SCALES[int(np.argmin(err))]

    loo_pred, loo_scale = [], []
    for i in range(len(tests)):
        others = [j for j in range(len(tests)) if j != i]
        s = best_scale(others)
        loo_scale.append(float(s))
        loo_pred.append(float(table[i, list(SCALES).index(s)]))
    loo_pred = np.array(loo_pred)
    s_all = float(best_scale(list(range(len(tests)))))

    mean_guess = np.array([np.mean(np.delete(meas, i)) for i in range(len(tests))])   # "always the average" reference
    out = dict(
        n_tests=len(tests),
        measured_mean_m=float(meas.mean()), measured_sd_m=float(meas.std()),
        uncalibrated=dict(mae_m=float(np.abs(base - meas).mean()), bias_m=float((base - meas).mean()),
                          mean_m=float(base.mean())),
        calibrated_leave_one_out=dict(mae_m=float(np.abs(loo_pred - meas).mean()),
                                      bias_m=float((loo_pred - meas).mean()),
                                      scales=loo_scale),
        average_of_other_tests=dict(mae_m=float(np.abs(mean_guess - meas).mean())),
        force_scale_all_tests=s_all,
        corr_mass_measured=float(np.corrcoef([t["test_mass_kg"] for t in tests], meas)[0, 1]),
        per_test=[dict(report=t["report_id"], vehicle=f"{t['year']} {t['vehicle']}", mass_kg=t["test_mass_kg"],
                       measured_m=float(m), layer_m=float(b), calibrated_loo_m=float(p))
                  for t, m, b, p in zip(tests, meas, base, loo_pred)],
    )
    print(f"{len(tests)} NHTSA full-frontal tests, measured maximum static crush {meas.mean():.3f} +- {meas.std():.3f} m")
    print(f"physics layer as published: mean {base.mean():.3f} m, MAE {out['uncalibrated']['mae_m']:.3f} m, "
          f"bias {out['uncalibrated']['bias_m']:+.3f} m")
    print(f"one force multiplier, leave-one-out: MAE {out['calibrated_leave_one_out']['mae_m']:.3f} m, "
          f"bias {out['calibrated_leave_one_out']['bias_m']:+.3f} m (multiplier {min(loo_scale)}-{max(loo_scale)}; "
          f"all tests: {s_all})")
    print(f"reference, average of the other tests: MAE {out['average_of_other_tests']['mae_m']:.3f} m")
    print(f"correlation of mass and measured crush: {out['corr_mass_measured']:+.2f}")
    for r in out["per_test"]:
        print(f"  {r['report']}  {r['vehicle']:38s} {r['mass_kg']:7.1f} kg  measured {r['measured_m']:.3f}  "
              f"layer {r['layer_m']:.3f}  calibrated {r['calibrated_loo_m']:.3f}")
    with open(os.path.join(HERE, "..", "results", "real_crash_tests.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
