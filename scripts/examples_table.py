"""Run every scenario in examples/ through the physics layer and print a markdown summary table.

    python scripts/examples_table.py [examples_dir]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from crashdamage.export import Predictor  # noqa: E402

folder = sys.argv[1] if len(sys.argv) > 1 else "examples"
predictor = Predictor()

rows = [
    "| example | speed (km/h) | crush depth (m) | cabin intrusion (m) | peak force (kN) | parts crushed or destroyed | torn off | obstacle | effects |",
    "|---|---:|---:|---:|---:|---:|---|---|---|",
]
for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
    with open(path) as f:
        scenario = json.load(f)
    rep = predictor.predict(scenario, engine="physics")
    s = rep["summary"]
    hard = sum(1 for p in rep["parts"] if p["level"] >= 3)
    torn = ", ".join(p["id"] for p in rep["parts"] if p["detached"]) or "-"
    obs = rep["obstacle"]["level_name"]
    if rep["obstacle"].get("broken_through"):
        obs += " (broke)"
    effects = ", ".join(k.replace("_", " ") for k, v in rep["effects"].items() if v) or "-"
    rows.append(f"| {os.path.splitext(os.path.basename(path))[0]} | {rep['impact']['speed_kmh']:.0f} | "
                f"{s['crush_depth_m']:.2f} | {s['cabin_intrusion_m']:.2f} | {s['peak_force_kn']:.0f} | "
                f"{hard} | {torn} | {obs} | {effects} |")
print("\n".join(rows))
