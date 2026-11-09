"""Training, prediction and saving of the neural models."""
from dataclasses import asdict, dataclass
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import Normalizer
from .models import build_model
from .physics import ELASTIC_FLOOR


@dataclass
class TrainConfig:
    kind: str = "gnn"                  # gnn | node_mlp | mlp
    hidden: int = 96
    layers: int = 3
    dropout: float = 0.0
    epochs: int = 60
    batch_size: int = 128
    lr: float = 2e-3
    weight_decay: float = 1e-5
    patience: int = 12                 # stop when the validation loss has not improved for this many epochs
    physics_features: bool = False     # add impact energy and part capacities as inputs
    energy_loss: float = 0.0           # weight of the energy-consistency penalty (0 = off)
    detach_pos_weight: float = 4.0     # parts rarely fall off, so the positive class is up-weighted
    node_weight: float = 5.0
    seed: int = 0


class Bundle:
    """A trained model together with everything needed to use it later."""

    def __init__(self, model, cfg, norm, reg_mean, reg_std, n_node_feat, n_glob_feat):
        self.model, self.cfg, self.norm = model, cfg, norm
        self.reg_mean, self.reg_std = reg_mean, reg_std
        self.n_node_feat, self.n_glob_feat = n_node_feat, n_glob_feat


def model_inputs(data, idx, cfg):
    nx, gx = data["node_x"][idx], data["glob_x"][idx]
    if cfg.physics_features:
        nx = np.concatenate([nx, data["phys_node"][idx]], axis=2)
        gx = np.concatenate([gx, data["phys_glob"][idx]], axis=1)
    return nx, gx


def _tensors(data, idx, cfg, norm, reg_mean, reg_std):
    nx, gx = model_inputs(data, idx, cfg)
    f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))
    return dict(node_x=f32(norm.node(nx)), glob_x=f32(norm.glob(gx)),
                damage=f32(data["damage"][idx]), detached=f32(data["detached"][idx]),
                reg=f32((data["glob_reg"][idx] - reg_mean) / reg_std), flags=f32(data["flags"][idx]),
                ecap=f32(data["ecap_kj"][idx]), e_diss=f32(data["e_diss_kj"][idx]))


def compute_loss(out, b, cfg):
    d = torch.sigmoid(out["damage"])
    l_node = F.mse_loss(d, b["damage"])
    l_det = F.binary_cross_entropy_with_logits(out["detach"], b["detached"],
                                               pos_weight=torch.tensor(cfg.detach_pos_weight))
    l_reg = F.smooth_l1_loss(out["reg"], b["reg"])
    l_flag = F.binary_cross_entropy_with_logits(out["flags"], b["flags"])
    base = cfg.node_weight * l_node + l_det + l_reg + l_flag
    # Energy consistency: the loaded parts cannot have absorbed more than the impact energy
    # (plus the small allowance the labels give to elastic marks).
    absorbed = (d * b["ecap"]).sum(dim=1)
    allowed = b["e_diss"] + ELASTIC_FLOOR * b["ecap"].sum(dim=1)
    l_energy = (F.relu(absorbed - allowed) / b["e_diss"].clamp(min=1.0)).mean()
    total = base + cfg.energy_loss * l_energy
    return total, base, dict(node=l_node.item(), detach=l_det.item(), reg=l_reg.item(), flags=l_flag.item(),
                             energy=l_energy.item())


def train(data, split, cfg, verbose=False):
    """Fit one model. ``split`` is (train_idx, val_idx, test_idx). Returns (Bundle, history)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    tr, va, _ = split
    nx, gx = model_inputs(data, tr, cfg)
    norm = Normalizer.fit(nx, gx)
    reg_mean = data["glob_reg"][tr].mean(axis=0)
    reg_std = data["glob_reg"][tr].std(axis=0) + 1e-6
    t_tr = _tensors(data, tr, cfg, norm, reg_mean, reg_std)
    t_va = _tensors(data, va, cfg, norm, reg_mean, reg_std)

    model = build_model(cfg.kind, nx.shape[-1], gx.shape[-1], cfg.hidden, cfg.layers, cfg.dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    n = len(tr)
    best, best_state, best_epoch = float("inf"), None, 0
    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n)
        run = 0.0
        for i in range(0, n, cfg.batch_size):
            sel = perm[i:i + cfg.batch_size]
            batch = {k: v[sel] for k, v in t_tr.items()}
            loss, _, _ = compute_loss(model(batch["node_x"], batch["glob_x"]), batch, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            run += loss.item() * len(sel)
        sched.step()
        model.eval()
        with torch.no_grad():
            _, val_base, val_parts = compute_loss(model(t_va["node_x"], t_va["glob_x"]), t_va, cfg)
        val = val_base.item()
        history.append(dict(epoch=epoch + 1, train_loss=run / n, val_loss=val, **val_parts))
        if verbose:
            print(f"epoch {epoch + 1:3d}  train {run / n:.4f}  val {val:.4f}  ({time.time() - t0:.0f}s)")
        if val < best - 1e-5:
            best, best_state, best_epoch = val, copy.deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    return Bundle(model, cfg, norm, reg_mean, reg_std, nx.shape[-1], gx.shape[-1]), history


@torch.no_grad()
def predict_arrays(bundle, node_x, glob_x, batch_size=1024):
    """node_x (n, parts, features) and glob_x (n, features), already including the physics
    features if the model was trained with them. Returns numpy arrays."""
    nx = torch.from_numpy(bundle.norm.node(node_x).astype(np.float32))
    gx = torch.from_numpy(bundle.norm.glob(glob_x).astype(np.float32))
    outs = []
    bundle.model.eval()
    for i in range(0, len(nx), batch_size):
        outs.append(bundle.model(nx[i:i + batch_size], gx[i:i + batch_size]))
    cat = lambda k: torch.cat([o[k] for o in outs]).numpy()
    return dict(damage=1 / (1 + np.exp(-cat("damage"))), detach_prob=1 / (1 + np.exp(-cat("detach"))),
                reg=cat("reg") * bundle.reg_std + bundle.reg_mean, flags_prob=1 / (1 + np.exp(-cat("flags"))))


def predict(bundle, data, idx):
    nx, gx = model_inputs(data, idx, bundle.cfg)
    return predict_arrays(bundle, nx, gx)


# -- saving -----------------------------------------------------------------------------

def save_bundle(path, b):
    torch.save(dict(cfg=asdict(b.cfg), state=b.model.state_dict(),
                    norm={k: torch.from_numpy(np.asarray(v)) for k, v in b.norm.state().items()},
                    reg_mean=torch.from_numpy(np.asarray(b.reg_mean)), reg_std=torch.from_numpy(np.asarray(b.reg_std)),
                    n_node_feat=b.n_node_feat, n_glob_feat=b.n_glob_feat), path)


def load_bundle(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = TrainConfig(**ck["cfg"])
    model = build_model(cfg.kind, ck["n_node_feat"], ck["n_glob_feat"], cfg.hidden, cfg.layers, cfg.dropout)
    model.load_state_dict(ck["state"])
    model.eval()
    norm = Normalizer.from_state({k: v.numpy() for k, v in ck["norm"].items()})
    return Bundle(model, cfg, norm, ck["reg_mean"].numpy(), ck["reg_std"].numpy(), ck["n_node_feat"], ck["n_glob_feat"])
