"""Neural models that predict the damage report of a collision.

Two models share the same inputs and outputs:

* ``GraphNet``  - works on the car's part graph. Every part is a node, edges are the
  load paths and fixings from ``vehicle.py``. Messages travel along the edges, so a hit
  on the door can change what is predicted for the pillar and the windscreen.
* ``FlatMLP``   - ignores the graph. It sees all parts and the scenario as one long vector.

Outputs per scenario
    node  : damage (0..1) and a "part falls off" logit for each of the parts
    global: crush depth, cabin intrusion, peak force, obstacle damage, and three
            yes/no flags (front airbag, side airbag, obstacle went through the car)
"""
import torch
from torch import nn

from .vehicle import N_PARTS, adjacency_matrix

N_GLOBAL_REG = 4      # crush depth, intrusion, peak force (MN), obstacle damage
N_FLAGS = 3           # airbag front, airbag side, went through the car


def _mlp(sizes, last_act=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_act:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class GraphNet(nn.Module):
    """Message passing on the fixed part graph (dense adjacency, no extra library needed)."""

    def __init__(self, n_node_feat, n_glob_feat, hidden=96, layers=3, dropout=0.0):
        super().__init__()
        a = torch.tensor(adjacency_matrix())
        a = a / a.sum(dim=1, keepdim=True).clamp(min=1e-6)      # rows add up to 1 (weighted mean of neighbours)
        self.register_buffer("adj", a)
        self.layers = layers
        self.node_in = _mlp([n_node_feat, hidden, hidden])
        self.glob_in = _mlp([n_glob_feat, hidden, hidden])
        self.msg = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(layers)])
        self.upd = nn.ModuleList([_mlp([3 * hidden, hidden, hidden]) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.node_out = _mlp([2 * hidden, hidden, 2])
        self.glob_out = _mlp([3 * hidden, hidden, N_GLOBAL_REG + N_FLAGS])

    def forward(self, node_x, glob_x):
        g = self.glob_in(glob_x)                                  # (B, H)
        h = self.node_in(node_x)                                  # (B, N, H)
        gn = g[:, None, :].expand(-1, h.shape[1], -1)
        for k in range(self.layers):
            m = torch.matmul(self.adj, self.msg[k](h))            # (B, N, H) weighted mean of neighbour messages
            h = self.norm[k](h + self.drop(self.upd[k](torch.cat([h, m, gn], dim=-1))))
        node = self.node_out(torch.cat([h, gn], dim=-1))          # (B, N, 2)
        pooled = torch.cat([h.mean(dim=1), h.max(dim=1).values, g], dim=-1)
        glob = self.glob_out(pooled)
        return dict(damage=node[..., 0], detach=node[..., 1],
                    reg=glob[:, :N_GLOBAL_REG], flags=glob[:, N_GLOBAL_REG:])


class FlatMLP(nn.Module):
    """Same inputs and outputs, but no knowledge of which part touches which."""

    def __init__(self, n_node_feat, n_glob_feat, hidden=256, layers=3, dropout=0.0):
        super().__init__()
        sizes = [N_PARTS * n_node_feat + n_glob_feat] + [hidden] * layers
        body = []
        for i in range(len(sizes) - 1):
            body += [nn.Linear(sizes[i], sizes[i + 1]), nn.GELU(), nn.Dropout(dropout)]
        self.body = nn.Sequential(*body)
        self.out = nn.Linear(hidden, N_PARTS * 2 + N_GLOBAL_REG + N_FLAGS)

    def forward(self, node_x, glob_x):
        b = node_x.shape[0]
        z = self.out(self.body(torch.cat([node_x.reshape(b, -1), glob_x], dim=-1)))
        node = z[:, :N_PARTS * 2].reshape(b, N_PARTS, 2)
        rest = z[:, N_PARTS * 2:]
        return dict(damage=node[..., 0], detach=node[..., 1],
                    reg=rest[:, :N_GLOBAL_REG], flags=rest[:, N_GLOBAL_REG:])


def build_model(kind, n_node_feat, n_glob_feat, hidden=None, layers=None, dropout=0.0):
    if kind == "gnn":
        return GraphNet(n_node_feat, n_glob_feat, hidden or 96, 3 if layers is None else layers, dropout)
    if kind == "node_mlp":        # the graph net with zero message-passing layers: every part on its own
        return GraphNet(n_node_feat, n_glob_feat, hidden or 96, 0, dropout)
    if kind == "mlp":
        return FlatMLP(n_node_feat, n_glob_feat, hidden or 256, 3 if layers is None else layers, dropout)
    raise ValueError(f"unknown model kind {kind!r}")
