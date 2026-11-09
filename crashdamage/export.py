"""Turn a scenario into the JSON damage report that the game engine reads.

Two engines can fill the report:

* ``physics`` - the transparent energy model (about a millisecond per scenario)
* ``model``   - a trained neural network that imitates the labels it was trained on

Both give the same JSON layout, so the Unity side does not care which one answered.
"""
import numpy as np

from .dataset import scenario_inputs
from .metrics import level_from_damage
from .physics import (MIN_OVERLAP, Result, Scenario, contact_geometry, effect_hints, impact_energy,
                      result_to_dict, scenario_from_dict, simulate)
from .train import load_bundle, predict_arrays


class Predictor:
    def __init__(self, model_path=None):
        self.bundle = load_bundle(model_path) if model_path else None

    @property
    def has_model(self):
        return self.bundle is not None

    def predict(self, scenario, engine="auto"):
        """scenario: a Scenario or the dict the game engine sends (see docs/method.md)."""
        sc = scenario if isinstance(scenario, Scenario) else scenario_from_dict(scenario)
        if engine == "auto":
            engine = "model" if self.has_model else "physics"
        if engine == "physics":
            report = result_to_dict(sc, simulate(sc))
        elif engine == "model":
            if not self.has_model:
                raise ValueError("no trained model is loaded; start with --model or use engine=physics")
            report = self._from_model(sc)
        else:
            raise ValueError(f"unknown engine {engine!r}")
        report["engine"] = engine
        return report

    def _from_model(self, sc):
        geo = contact_geometry(sc)
        inp = scenario_inputs(sc, geo)
        nx, gx = inp["node_x"][None], inp["glob_x"][None]
        if self.bundle.cfg.physics_features:
            nx = np.concatenate([nx, inp["phys_node"][None]], axis=2)
            gx = np.concatenate([gx, inp["phys_glob"][None]], axis=1)
        out = predict_arrays(self.bundle, nx, gx)

        damage = np.clip(out["damage"][0], 0.0, 1.0)
        damage = np.where(damage < 0.01, 0.0, damage)
        crush, intrusion, peak_mn, obstacle_damage = (float(v) for v in out["reg"][0])
        obstacle_damage = float(np.clip(obstacle_damage, 0.0, 1.0))
        airbag_front, airbag_side, through = (bool(p >= 0.5) for p in out["flags_prob"][0])
        e_diss, dv, v_obs, _ = impact_energy(sc, geo)

        res = Result(damage=damage, level=level_from_damage(damage), detached=out["detach_prob"][0] >= 0.5,
                     direct=geo["overlap"] > MIN_OVERLAP, crush_depth=max(crush, 0.0), intrusion=max(intrusion, 0.0),
                     peak_force_kn=max(peak_mn, 0.0) * 1000.0, e_diss_kj=e_diss / 1e3, e_car_kj=0.0, e_obstacle_kj=0.0,
                     obstacle_damage=obstacle_damage, obstacle_level=int(level_from_damage(obstacle_damage)),
                     obstacle_depth=0.0, breakthrough=False, airbag_front=airbag_front, airbag_side=airbag_side,
                     delta_v=dv, obstacle_speed_after=v_obs, through=through)
        res.effects = effect_hints(sc, res)
        report = result_to_dict(sc, res)
        # these are not predicted by the network, so they are left out instead of being reported as zero
        for key in ("energy_car_kj", "energy_obstacle_kj", "energy_unabsorbed_kj"):
            report["summary"].pop(key, None)
        for key in ("deformation_m", "broken_through"):
            report["obstacle"].pop(key, None)
        return report
