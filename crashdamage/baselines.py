"""Models that are not neural networks, used as reference points.

* ``NaiveEnergyBaseline``  - no learning. Spreads the impact energy over the loaded parts in
  proportion to what they can absorb. This is what you get from a back-of-the-envelope
  energy balance without ordering, obstacle give or load paths.
* ``GradientBoostingBaseline`` - scikit-learn histogram gradient boosting on the same inputs.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .vehicle import N_PARTS


class NaiveEnergyBaseline:
    name = "naive_energy"

    def fit(self, data, idx):
        return self

    def predict(self, data, idx):
        e = data["e_diss_kj"][idx]
        cap = data["ecap_kj"][idx]
        total = cap.sum(axis=1, keepdims=True)
        share = np.where(total > 0, np.minimum(1.0, e[:, None] / np.maximum(total, 1e-9)), 0.0)
        return dict(damage=np.where(cap > 0, share, 0.0))


class GradientBoostingBaseline:
    name = "gbm"

    def __init__(self, seed=0, max_iter=150, physics_features=False):
        self.seed, self.max_iter, self.physics = seed, max_iter, physics_features

    # -- input layouts ---------------------------------------------------------------
    def _long(self, data, idx):
        """One row per (scenario, part)."""
        nx, gx = data["node_x"][idx], data["glob_x"][idx]
        n = len(idx)
        cols = [nx.reshape(n * N_PARTS, -1), np.repeat(gx, N_PARTS, axis=0), np.tile(np.arange(N_PARTS), n)[:, None]]
        if self.physics:
            cols += [data["phys_node"][idx].reshape(n * N_PARTS, -1), np.repeat(data["phys_glob"][idx], N_PARTS, axis=0)]
        return np.concatenate(cols, axis=1)

    def _flat(self, data, idx):
        """One row per scenario."""
        n = len(idx)
        cols = [data["node_x"][idx].reshape(n, -1), data["glob_x"][idx]]
        if self.physics:
            cols += [data["phys_node"][idx].reshape(n, -1), data["phys_glob"][idx]]
        return np.concatenate(cols, axis=1)

    def _reg(self, **kw):
        return HistGradientBoostingRegressor(max_iter=self.max_iter, learning_rate=0.1, random_state=self.seed, **kw)

    def _clf(self, **kw):
        return HistGradientBoostingClassifier(max_iter=self.max_iter, learning_rate=0.1, random_state=self.seed, **kw)

    # -- fit / predict ---------------------------------------------------------------
    def fit(self, data, idx):
        xl = self._long(data, idx)
        self.m_damage = self._reg().fit(xl, data["damage"][idx].reshape(-1))
        self.m_detach = self._clf(class_weight="balanced").fit(xl, data["detached"][idx].reshape(-1).astype(int))
        xf = self._flat(data, idx)
        self.m_reg = [self._reg(max_features=0.3).fit(xf, data["glob_reg"][idx][:, k]) for k in range(4)]
        self.m_flags = []
        for k in range(3):
            y = data["flags"][idx][:, k].astype(int)
            self.m_flags.append(self._clf(max_features=0.3).fit(xf, y) if len(np.unique(y)) > 1 else int(y[0]))
        return self

    def predict(self, data, idx):
        n = len(idx)
        xl, xf = self._long(data, idx), self._flat(data, idx)
        damage = np.clip(self.m_damage.predict(xl), 0, 1).reshape(n, N_PARTS)
        detach = self.m_detach.predict_proba(xl)[:, 1].reshape(n, N_PARTS)
        reg = np.stack([m.predict(xf) for m in self.m_reg], axis=1)
        flags = np.stack([m.predict_proba(xf)[:, 1] if not isinstance(m, int) else np.full(n, float(m))
                          for m in self.m_flags], axis=1)
        return dict(damage=damage, detach_prob=detach, reg=reg, flags_prob=flags)
