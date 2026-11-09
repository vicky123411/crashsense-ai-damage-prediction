import numpy as np
import pytest

from crashdamage import dataset
from crashdamage.external import load_fe_dataset, write_csv
from crashdamage.features import (GLOBAL_FEATURE_NAMES, NODE_FEATURE_NAMES, PHYS_GLOBAL_NAMES, PHYS_NODE_NAMES,
                                  global_features, node_features, physics_features)
from crashdamage.obstacles import PRESET_NAMES
from crashdamage.physics import simulate
from crashdamage.vehicle import FACES, N_PARTS


def test_feature_shapes_and_values(rng):
    sc = dataset.random_scenario(rng)
    nx, gx = node_features(sc), global_features(sc)
    assert nx.shape == (N_PARTS, len(NODE_FEATURE_NAMES))
    assert gx.shape == (len(GLOBAL_FEATURE_NAMES),)
    pn, pg = physics_features(sc)
    assert pn.shape == (N_PARTS, len(PHYS_NODE_NAMES)) and pg.shape == (len(PHYS_GLOBAL_NAMES),)
    for a in (nx, gx, pn, pg):
        assert np.isfinite(a).all()


def test_generation_is_repeatable():
    a, b = dataset.generate(25, seed=3), dataset.generate(25, seed=3)
    for k in ("node_x", "glob_x", "damage", "glob_reg", "flags"):
        assert np.array_equal(a[k], b[k])
    assert not np.array_equal(a["damage"], dataset.generate(25, seed=4)["damage"])


def test_labels_match_the_physics_layer(rng):
    rng2 = np.random.default_rng(9)
    data = dataset.generate(10, seed=9)
    for i in range(10):
        res = simulate(dataset.random_scenario(rng2))
        assert np.allclose(data["damage"][i], res.damage, atol=1e-6)
        assert data["glob_reg"][i][0] == pytest.approx(res.crush_depth, abs=1e-5)


def test_dataset_has_all_kinds_of_outcomes(small_data):
    assert len(np.unique(small_data["level"])) == 5
    assert small_data["detached"].sum() > 0
    assert 0 < small_data["flags"][:, 2].mean() < 0.5
    assert set(np.unique(small_data["preset"])) == set(range(len(PRESET_NAMES)))


@pytest.mark.parametrize("kind", ["random", "speed", "obstacle", "face"])
def test_splits_do_not_overlap(small_data, kind):
    tr, va, te = dataset.make_split(small_data, kind, seed=1)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    assert len(tr) > 0 and len(va) > 0 and len(te) > 0


def test_hold_out_splits_really_hold_out(small_data):
    tr, _, te = dataset.make_split(small_data, "speed", speed_cut=24.0)
    assert small_data["speed"][tr].max() < 24.0 <= small_data["speed"][te].min()
    tr, _, te = dataset.make_split(small_data, "obstacle", holdout="steel_crate")
    k = PRESET_NAMES.index("steel_crate")
    assert (small_data["preset"][tr] != k).all() and (small_data["preset"][te] == k).all()
    tr, _, te = dataset.make_split(small_data, "face", holdout_face="rear")
    k = FACES.index("rear")
    assert (small_data["face"][tr] != k).all() and (small_data["face"][te] == k).all()


def test_normalizer_uses_training_rows_only(small_data):
    tr, _, _ = dataset.make_split(small_data, "random")
    norm = dataset.Normalizer.fit(small_data["node_x"][tr], small_data["glob_x"][tr])
    z = norm.node(small_data["node_x"][tr]).reshape(-1, small_data["node_x"].shape[-1])
    assert np.allclose(z.mean(0), 0, atol=1e-3)
    again = dataset.Normalizer.from_state(norm.state())
    assert np.allclose(again.node_mean, norm.node_mean)


def test_external_labels_round_trip(tmp_path):
    rng = np.random.default_rng(2)
    scenarios = [dataset.random_scenario(rng) for _ in range(12)]
    results = [simulate(s) for s in scenarios]
    write_csv(scenarios, results, tmp_path / "cases.csv", tmp_path / "parts.csv")
    data = load_fe_dataset(tmp_path / "cases.csv", tmp_path / "parts.csv")
    assert str(data["source"]) == "external_fe"
    expected = [dataset.scenario_inputs(s) for s in scenarios]
    assert np.allclose(data["node_x"], np.stack([e["node_x"] for e in expected]), rtol=1e-4, atol=1e-5)
    assert np.allclose(data["glob_x"], np.stack([e["glob_x"] for e in expected]), rtol=1e-4, atol=1e-5)
    assert np.allclose(data["damage"], np.stack([r.damage for r in results]), atol=1e-6)
    assert np.array_equal(data["detached"], np.stack([r.detached for r in results]).astype(np.float32))


def test_external_reader_rejects_unknown_parts(tmp_path):
    rng = np.random.default_rng(2)
    sc = dataset.random_scenario(rng)
    write_csv([sc], [simulate(sc)], tmp_path / "cases.csv", tmp_path / "parts.csv")
    text = (tmp_path / "parts.csv").read_text().replace("door_l", "door_x")
    (tmp_path / "parts.csv").write_text(text)
    with pytest.raises(ValueError, match="unknown part_id"):
        load_fe_dataset(tmp_path / "cases.csv", tmp_path / "parts.csv")
