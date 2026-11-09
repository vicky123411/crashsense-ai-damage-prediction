"""Random scenarios, labels from the physics layer, and train / test splits."""
import numpy as np

from .features import global_features, node_features, physics_features
from .metrics import level_from_damage
from .obstacles import PRESET_NAMES, sample_obstacle
from .physics import Scenario, contact_geometry, simulate
from .vehicle import FACES, sample_vehicle

FACE_PROBS = [0.45, 0.10, 0.225, 0.225]


def random_scenario(rng, preset=None, face=None):
    vehicle = sample_vehicle(rng)
    obstacle = sample_obstacle(rng, preset)
    face = face or str(rng.choice(FACES, p=FACE_PROBS))
    return Scenario(vehicle, obstacle, face=face,
                    offset=float(rng.uniform(-1, 1)),
                    angle_deg=float(np.clip(rng.normal(0, 12), -40, 40)),
                    speed=float(rng.uniform(4, 36)))


def scenario_inputs(sc, geo=None):
    """Model inputs of one scenario (nothing here comes out of the solver)."""
    geo = geo or contact_geometry(sc)
    phys_node, phys_glob = physics_features(sc, geo)
    return dict(node_x=node_features(sc, geo), glob_x=global_features(sc, geo),
                phys_node=phys_node, phys_glob=phys_glob)


def generate(n, seed=0, source="physics_layer"):
    """Make n random crashes and label them. Returns a dict of numpy arrays."""
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ["node_x", "glob_x", "phys_node", "phys_glob", "damage", "level", "detached", "direct",
                           "ecap_kj", "e_diss_kj", "glob_reg", "obstacle_level", "flags", "speed", "face", "preset"]}
    for _ in range(n):
        sc = random_scenario(rng)
        res = simulate(sc)
        for k, v in scenario_inputs(sc, res.geo).items():
            out[k].append(v)
        out["damage"].append(res.damage)
        out["level"].append(res.level)
        out["detached"].append(res.detached.astype(np.float32))
        out["direct"].append(res.direct.astype(np.float32))
        out["ecap_kj"].append(res.capacity_kj)
        out["e_diss_kj"].append(res.e_diss_kj)
        out["glob_reg"].append([res.crush_depth, res.intrusion, res.peak_force_kn / 1000.0, res.obstacle_damage])
        out["obstacle_level"].append(res.obstacle_level)
        out["flags"].append([float(res.airbag_front), float(res.airbag_side), float(res.through)])
        out["speed"].append(sc.speed)
        out["face"].append(FACES.index(sc.face))
        out["preset"].append(PRESET_NAMES.index(sc.obstacle.name))
    data = {k: np.array(v) for k, v in out.items()}
    for k in ("node_x", "glob_x", "phys_node", "phys_glob", "damage", "detached", "direct", "ecap_kj", "glob_reg", "flags"):
        data[k] = data[k].astype(np.float32)
    for k in ("level", "obstacle_level", "face", "preset"):
        data[k] = data[k].astype(np.int64)
    data["e_diss_kj"] = data["e_diss_kj"].astype(np.float32)
    data["speed"] = data["speed"].astype(np.float32)
    # levels are taken from the stored (float32) damage, so labels and thresholds always agree
    data["level"] = level_from_damage(data["damage"]).astype(np.int64)
    data["obstacle_level"] = level_from_damage(data["glob_reg"][:, 3]).astype(np.int64)
    data["source"] = np.array(source)
    return data


def save(path, data):
    np.savez_compressed(path, **data)


def load(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def make_split(data, kind="random", seed=0, holdout="steel_crate", speed_cut=24.0, holdout_face="rear"):
    """Index arrays for train / val / test.

    random   - ordinary random split (in-distribution)
    speed    - train on slower crashes, test on faster ones than anything seen in training
    obstacle - one obstacle type never appears in training
    face     - one side of the car is never hit in training
    """
    n = len(data["speed"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    if kind == "random":
        a, b = int(0.7 * n), int(0.85 * n)
        return idx[:a], idx[a:b], idx[b:]
    if kind == "speed":
        test = np.where(data["speed"] >= speed_cut)[0]
        rest = np.array([i for i in idx if data["speed"][i] < speed_cut])
    elif kind == "obstacle":
        k = PRESET_NAMES.index(holdout)
        test = np.where(data["preset"] == k)[0]
        rest = np.array([i for i in idx if data["preset"][i] != k])
    elif kind == "face":
        k = FACES.index(holdout_face)
        test = np.where(data["face"] == k)[0]
        rest = np.array([i for i in idx if data["face"][i] != k])
    else:
        raise ValueError(f"unknown split {kind}")
    n_val = max(1, int(0.15 * len(rest)))
    return rest[n_val:], rest[:n_val], test


class Normalizer:
    """Mean / std computed on the training set only."""

    def __init__(self, node_mean, node_std, glob_mean, glob_std):
        self.node_mean, self.node_std = node_mean, node_std
        self.glob_mean, self.glob_std = glob_mean, glob_std

    @staticmethod
    def _scale(std):
        # A column that never changes in the training rows (for example the one-hot of a face that is held
        # out) is left unscaled. Dividing it by ~1e-6 would turn a 1 at test time into 1,000,000.
        return np.where(std < 1e-6, 1.0, std + 1e-6)

    @classmethod
    def fit(cls, node_x, glob_x):
        """node_x: (n, parts, features), glob_x: (n, features), both from the training rows."""
        nx = node_x.reshape(-1, node_x.shape[-1])
        return cls(nx.mean(0), cls._scale(nx.std(0)), glob_x.mean(0), cls._scale(glob_x.std(0)))

    def node(self, x):
        return (x - self.node_mean) / self.node_std

    def glob(self, x):
        return (x - self.glob_mean) / self.glob_std

    def state(self):
        return dict(node_mean=self.node_mean, node_std=self.node_std, glob_mean=self.glob_mean, glob_std=self.glob_std)

    @classmethod
    def from_state(cls, s):
        return cls(np.asarray(s["node_mean"]), np.asarray(s["node_std"]), np.asarray(s["glob_mean"]), np.asarray(s["glob_std"]))
