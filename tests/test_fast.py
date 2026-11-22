"""The NumPy fast path must give the same answers as the PyTorch model."""
import json
import os

import numpy as np
import pytest

from crashdamage import dataset
from crashdamage.export import Predictor
from crashdamage.features import IMPACT_GLOBAL_NAMES, IMPACT_NODE_NAMES

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "gnn_demo.pt")


@pytest.mark.skipif(not os.path.exists(MODEL), reason="demo model not present")
def test_fast_matches_torch():
    fast, slow = Predictor(MODEL), Predictor(MODEL, fast=False)
    assert fast.fast is not None and slow.fast is None
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(60):
        sc = dataset.random_scenario(rng)
        a, b = fast.predict(sc, "model"), slow.predict(sc, "model")
        da = np.array([p["damage"] for p in a["parts"]])
        db = np.array([p["damage"] for p in b["parts"]])
        worst = max(worst, float(np.abs(da - db).max()))
        assert [p["detached"] for p in a["parts"]] == [p["detached"] for p in b["parts"]]
        assert a["effects"] == b["effects"]
    assert worst <= 2e-4          # report values are rounded to 4 decimals; float32 may move the last one


def test_impact_frame_features_shape():
    rng = np.random.default_rng(3)
    sc = dataset.random_scenario(rng)
    inp = dataset.scenario_inputs(sc)
    assert inp["node_xi"].shape == (28, len(IMPACT_NODE_NAMES))
    assert inp["glob_xi"].shape == (len(IMPACT_GLOBAL_NAMES),)
    # the impact frame has no "which side was hit" one-hot
    assert not any(n in ("face_front", "face_rear", "face_left", "face_right") for n in IMPACT_GLOBAL_NAMES)


def test_impact_frame_depth_is_from_the_hit_point():
    from crashdamage.obstacles import default_obstacle
    from crashdamage.physics import Scenario
    from crashdamage.vehicle import PART_INDEX, default_vehicle
    v, ob = default_vehicle(), default_obstacle("concrete_wall")
    col = IMPACT_NODE_NAMES.index("depth_from_hit")
    front = dataset.scenario_inputs(Scenario(v, ob, face="front", offset=0, angle_deg=0, speed=10))["node_xi"]
    rear = dataset.scenario_inputs(Scenario(v, ob, face="rear", offset=0, angle_deg=0, speed=10))["node_xi"]
    # the front bumper is near the hit point in a front crash and far from it in a rear crash, and vice versa
    assert front[PART_INDEX["bumper_front"], col] < 0.2 < rear[PART_INDEX["bumper_front"], col]
    assert rear[PART_INDEX["bumper_rear"], col] < 0.2 < front[PART_INDEX["bumper_rear"], col]
