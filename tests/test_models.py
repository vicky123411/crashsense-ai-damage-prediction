import numpy as np
import pytest
import torch

from crashdamage import dataset
from crashdamage.baselines import GradientBoostingBaseline, NaiveEnergyBaseline
from crashdamage.export import Predictor
from crashdamage.features import N_GLOBAL_FEATURES, N_NODE_FEATURES
from crashdamage.metrics import evaluate_predictions, level_from_damage, macro_f1
from crashdamage.models import N_FLAGS, N_GLOBAL_REG, build_model
from crashdamage.train import TrainConfig, load_bundle, predict, save_bundle, train
from crashdamage.vehicle import N_PARTS


@pytest.mark.parametrize("kind", ["gnn", "node_mlp", "mlp"])
def test_forward_shapes(kind):
    model = build_model(kind, N_NODE_FEATURES, N_GLOBAL_FEATURES, hidden=16)
    out = model(torch.randn(5, N_PARTS, N_NODE_FEATURES), torch.randn(5, N_GLOBAL_FEATURES))
    assert out["damage"].shape == (5, N_PARTS) and out["detach"].shape == (5, N_PARTS)
    assert out["reg"].shape == (5, N_GLOBAL_REG) and out["flags"].shape == (5, N_FLAGS)


def test_graph_net_lets_information_travel_between_parts():
    torch.manual_seed(0)
    model = build_model("gnn", N_NODE_FEATURES, N_GLOBAL_FEATURES, hidden=16, layers=2).eval()
    zero_layers = build_model("node_mlp", N_NODE_FEATURES, N_GLOBAL_FEATURES, hidden=16).eval()
    x, g = torch.randn(1, N_PARTS, N_NODE_FEATURES), torch.randn(1, N_GLOBAL_FEATURES)
    x2 = x.clone()
    x2[0, 0] += 5.0                       # change only part 0
    with torch.no_grad():
        d_graph = (model(x, g)["damage"] - model(x2, g)["damage"])[0, 1:].abs().max()
        d_flat = (zero_layers(x, g)["damage"] - zero_layers(x2, g)["damage"])[0, 1:].abs().max()
    assert d_graph > 1e-6       # other parts react
    assert d_flat < 1e-6        # without message passing they do not


def test_training_reduces_the_loss(small_data):
    split = dataset.make_split(small_data, "random")
    cfg = TrainConfig(kind="gnn", hidden=24, layers=2, epochs=6, batch_size=64, seed=0)
    bundle, hist = train(small_data, split, cfg)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    out = predict(bundle, small_data, split[2])
    assert out["damage"].shape == (len(split[2]), N_PARTS)
    assert np.all((out["damage"] >= 0) & (out["damage"] <= 1))


def test_energy_loss_and_physics_features_run(small_data):
    split = dataset.make_split(small_data, "random")
    cfg = TrainConfig(kind="gnn", hidden=16, layers=1, epochs=2, physics_features=True, energy_loss=1.0)
    bundle, hist = train(small_data, split, cfg)
    assert bundle.n_node_feat == N_NODE_FEATURES + 2 and bundle.n_glob_feat == N_GLOBAL_FEATURES + 3
    assert "energy" in hist[0]


def test_saved_model_gives_the_same_predictions(small_data, tmp_path):
    split = dataset.make_split(small_data, "random")
    bundle, _ = train(small_data, split, TrainConfig(kind="gnn", hidden=16, layers=1, epochs=2))
    path = tmp_path / "m.pt"
    save_bundle(path, bundle)
    again = load_bundle(path)
    a, b = predict(bundle, small_data, split[2]), predict(again, small_data, split[2])
    for k in a:
        assert np.allclose(a[k], b[k], atol=1e-5)


def test_predictor_serves_both_engines(small_data, tmp_path):
    split = dataset.make_split(small_data, "random")
    bundle, _ = train(small_data, split, TrainConfig(kind="gnn", hidden=16, layers=1, epochs=2, physics_features=True))
    path = tmp_path / "m.pt"
    save_bundle(path, bundle)
    request = {"speed_kmh": 45, "face": "left", "offset": 0.1, "obstacle": {"name": "steel_pole"}}
    with_model = Predictor(path).predict(request)
    assert with_model["engine"] == "model" and len(with_model["parts"]) == N_PARTS
    assert "energy_car_kj" not in with_model["summary"]
    assert Predictor(path).predict(request, engine="physics")["engine"] == "physics"
    assert Predictor().predict(request)["engine"] == "physics"
    with pytest.raises(ValueError):
        Predictor().predict(request, engine="model")


def test_baselines_and_metrics_run(small_data):
    split = dataset.make_split(small_data, "random")
    naive = NaiveEnergyBaseline().predict(small_data, split[2])
    m = evaluate_predictions(small_data, split[2], naive)
    assert 0 <= m["damage_mae"] <= 1
    gb = GradientBoostingBaseline(max_iter=10).fit(small_data, split[0])
    m = evaluate_predictions(small_data, split[2], gb.predict(small_data, split[2]))
    assert {"damage_mae", "detach_f1", "crush_depth_m_mae", "through_f1"} <= set(m)


def test_perfect_predictions_score_perfectly(small_data):
    idx = np.arange(50)
    perfect = dict(damage=small_data["damage"][idx], detach_prob=small_data["detached"][idx],
                   reg=small_data["glob_reg"][idx], flags_prob=small_data["flags"][idx])
    m = evaluate_predictions(small_data, idx, perfect)
    assert m["damage_mae"] == 0 and m["level_acc"] == 1.0 and m["detach_f1"] == 1.0
    assert m["crush_depth_m_mae"] == 0 and m["through_f1"] == 1.0


def test_level_and_f1_helpers():
    assert list(level_from_damage(np.array([0.0, 0.049, 0.05, 0.2, 0.5, 0.85, 1.0]))) == [0, 0, 1, 2, 3, 4, 4]
    assert macro_f1(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0
    assert macro_f1(np.array([0, 0, 1]), np.array([1, 1, 1])) < 1.0
