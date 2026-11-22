"""Sanity check: physics-layer crush depth against a published crash-test stiffness model.

The CRASH3 crush-energy model, fitted to NHTSA full-width rigid-barrier tests at about 56 km/h,
says that a car with frontal stiffness coefficients A (kgf/cm) and B (kgf/cm^2) absorbs, per cm
of crush width w,

    E = w * (A*C + B*C^2/2 + A^2/(2B))          (kgf*cm, C = average residual crush in cm)

Kodsi et al. (2017, Accident Reconstruction Journal) report the class averages for cars of
800-1250 kg: A = 53.15 +- 11.12 kg/cm, B = 7.11 +- 2.49 kg/cm^2. This script solves the
model for C at 56 km/h for the same car and compares it with the physics layer's crush depth.
It is a rough plausibility check, not a validation: CRASH3 gives residual crush after spring-back,
the physics layer gives the deepest crush during the impact.

Two NHTSA NCAP full-frontal rigid-barrier reports (about 56.3 km/h) are added as measured points:
2023 Kia Niro, 1634.5 kg, maximum static crush 623 mm (NHTSA test report v14316R001);
2025 Mitsubishi Outlander, 1926.5 kg, maximum static crush 473 mm (NHTSA test report v15456R001).

    python scripts/crash_test_check.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crashdamage.obstacles import default_obstacle  # noqa: E402
from crashdamage.physics import Scenario, simulate  # noqa: E402
from crashdamage.vehicle import default_vehicle  # noqa: E402

G = 9.81
A_MEAN, A_SD = 53.15, 11.12      # kgf/cm, cars 800-1250 kg (Kodsi et al. 2017)
B_MEAN, B_SD = 7.11, 2.49        # kgf/cm^2
WIDTH_CM = 150.0                 # crush width: the bumper beam (1.50 m) spans the struck front


def crash3_crush_m(energy_j, a, b, width_cm=WIDTH_CM):
    """Average residual crush (m) that absorbs energy_j according to CRASH3."""
    e_kgfcm = energy_j / (G * 0.01)          # J -> kgf*cm
    g0 = a * a / (2 * b)
    # w*(b/2*C^2 + a*C + g0) = e  ->  quadratic in C (cm)
    qa, qb, qc = width_cm * b / 2, width_cm * a, width_cm * g0 - e_kgfcm
    c_cm = (-qb + np.sqrt(qb * qb - 4 * qa * qc)) / (2 * qa)
    return c_cm / 100.0


def main():
    rows = []
    for mass in (1000.0, 1100.0, 1250.0, 1634.5, 1926.5):
        sc = Scenario(default_vehicle(mass), default_obstacle("concrete_wall"), face="front", offset=0.0,
                      angle_deg=0.0, speed=56 / 3.6)
        res = simulate(sc)
        e = 0.5 * mass * (56 / 3.6) ** 2
        mid = crash3_crush_m(e, A_MEAN, B_MEAN)
        # stiffer car (A, B one SD up) crushes less; softer car more
        lo = crash3_crush_m(e, A_MEAN + A_SD, B_MEAN + B_SD)
        hi = crash3_crush_m(e, A_MEAN - A_SD, B_MEAN - B_SD)
        rows.append((mass, res.crush_depth, mid, lo, hi))
    print("full-width rigid wall, 56 km/h, front, no angle")
    print(f"{'mass kg':>8} {'physics layer m':>16} {'CRASH3 mean m':>14} {'CRASH3 range (+-1 SD) m':>24}")
    for mass, ours, mid, lo, hi in rows:
        if mass <= 1250:                      # the A, B averages are for cars of 800-1250 kg only
            print(f"{mass:8.0f} {ours:16.2f} {mid:14.2f} {lo:11.2f} to {hi:5.2f}")
        else:
            print(f"{mass:8.0f} {ours:16.2f} {'-':>14} {'-':>24}")
    print("\nmeasured (NHTSA NCAP, 56.3 km/h, maximum static crush): Kia Niro 2023, 1634.5 kg: 0.62 m;"
          " Mitsubishi Outlander 2025, 1926.5 kg: 0.47 m")
    return rows


if __name__ == "__main__":
    main()
