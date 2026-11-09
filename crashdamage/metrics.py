"""Scores used to compare the models. Everything is computed against the labels of the
dataset the models were trained on (the physics layer, or labels from an external solver)."""
import numpy as np

from .physics import LEVEL_EDGES

REG_NAMES = ["crush_depth_m", "intrusion_m", "peak_force_mn", "obstacle_damage"]
FLAG_NAMES = ["airbag_front", "airbag_side", "through"]


def level_from_damage(d):
    return np.searchsorted(LEVEL_EDGES, d, side="right")


def _f1(y, p):
    tp = float(np.sum((y == 1) & (p == 1)))
    fp = float(np.sum((y == 0) & (p == 1)))
    fn = float(np.sum((y == 1) & (p == 0)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def macro_f1(y, p, n_classes=5):
    scores = []
    for c in range(n_classes):
        if np.any(y == c) or np.any(p == c):
            scores.append(_f1((y == c).astype(int), (p == c).astype(int))[2])
    return float(np.mean(scores)) if scores else 0.0


def evaluate_predictions(data, idx, pred):
    """Compare predictions with the labels of the rows ``idx``.

    ``pred`` may hold: damage (n, N), detach_prob (n, N), reg (n, 4), flags_prob (n, 3).
    Only the parts that are present are scored.
    """
    m = {}
    if "damage" in pred:
        d_true, d_pred = data["damage"][idx], np.clip(pred["damage"], 0, 1)
        err = d_pred - d_true
        m["damage_mae"] = float(np.abs(err).mean())
        m["damage_rmse"] = float(np.sqrt((err ** 2).mean()))
        var = float(((d_true - d_true.mean()) ** 2).mean())
        m["damage_r2"] = float(1.0 - (err ** 2).mean() / var) if var > 0 else 0.0
        loaded = data["ecap_kj"][idx] > 0
        m["damage_mae_loaded"] = float(np.abs(err[loaded]).mean()) if loaded.any() else 0.0
        m["damage_mae_unloaded"] = float(np.abs(err[~loaded]).mean()) if (~loaded).any() else 0.0
        lv_true, lv_pred = data["level"][idx].ravel(), level_from_damage(d_pred).ravel()
        m["level_acc"] = float((lv_true == lv_pred).mean())
        m["level_within_one"] = float((np.abs(lv_true - lv_pred) <= 1).mean())
        m["level_macro_f1"] = macro_f1(lv_true, lv_pred)
    if "detach_prob" in pred:
        y = data["detached"][idx].ravel().astype(int)
        p = (pred["detach_prob"].ravel() >= 0.5).astype(int)
        m["detach_precision"], m["detach_recall"], m["detach_f1"] = _f1(y, p)
    if "reg" in pred:
        reg_true = data["glob_reg"][idx]
        for k, name in enumerate(REG_NAMES):
            m[f"{name}_mae"] = float(np.abs(pred["reg"][:, k] - reg_true[:, k]).mean())
        lv_true = level_from_damage(reg_true[:, 3])
        lv_pred = level_from_damage(np.clip(pred["reg"][:, 3], 0, 1))
        m["obstacle_level_acc"] = float((lv_true == lv_pred).mean())
    if "flags_prob" in pred:
        y = data["flags"][idx].astype(int)
        p = (pred["flags_prob"] >= 0.5).astype(int)
        for k, name in enumerate(FLAG_NAMES):
            m[f"{name}_acc"] = float((y[:, k] == p[:, k]).mean())
            m[f"{name}_f1"] = _f1(y[:, k], p[:, k])[2]
    return m


# the handful of numbers that go into the summary tables
HEADLINE = ["damage_mae", "damage_r2", "level_macro_f1", "detach_f1", "crush_depth_m_mae",
            "obstacle_damage_mae", "through_f1"]
