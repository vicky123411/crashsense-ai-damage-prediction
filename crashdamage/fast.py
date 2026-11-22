"""Fast single-request prediction without PyTorch.

For one scenario at a time, most of the time in PyTorch goes to calling into the library
(dozens of small layers, each with its own overhead), not to the arithmetic. This module copies
the trained weights of a ``GraphNet`` into NumPy arrays once, and then runs the same forward
pass with a handful of NumPy calls. The results match the PyTorch model to float32 rounding
(tested in tests/test_fast.py).
"""
import numpy as np
from scipy.special import erf

from .models import N_GLOBAL_REG, GraphNet


def _gelu(x):
    return 0.5 * x * (1.0 + erf(x * 0.7071067811865476))


def _linears(seq):
    """(weight.T, bias) of every Linear layer in an nn.Sequential built by models._mlp."""
    out = []
    for m in seq:
        if hasattr(m, "weight"):
            out.append((m.weight.detach().numpy().T.astype(np.float32).copy(),
                        m.bias.detach().numpy().astype(np.float32).copy()))
    return out


def _run(layers, x):
    for i, (w, b) in enumerate(layers):
        x = x @ w + b
        if i < len(layers) - 1:
            x = _gelu(x)
    return x


class FastGraphNet:
    def __init__(self, bundle):
        m = bundle.model
        if not isinstance(m, GraphNet):
            raise TypeError("FastGraphNet only supports the graph network (and the per-part MLP)")
        self.adj = m.adj.numpy().astype(np.float32)
        self.node_in, self.glob_in = _linears(m.node_in), _linears(m.glob_in)
        self.msg = [(l.weight.detach().numpy().T.astype(np.float32).copy(), l.bias.detach().numpy().astype(np.float32))
                    for l in m.msg]
        self.upd = [_linears(u) for u in m.upd]
        self.norm = [(n.weight.detach().numpy().astype(np.float32), n.bias.detach().numpy().astype(np.float32), n.eps)
                     for n in m.norm]
        self.node_out, self.glob_out = _linears(m.node_out), _linears(m.glob_out)
        nrm = bundle.norm
        self.nm, self.ns = nrm.node_mean.astype(np.float32), nrm.node_std.astype(np.float32)
        self.gm, self.gs = nrm.glob_mean.astype(np.float32), nrm.glob_std.astype(np.float32)
        self.reg_mean, self.reg_std = bundle.reg_mean, bundle.reg_std

    def forward(self, node_x, glob_x):
        """node_x (parts, features), glob_x (features,) -> raw outputs for one scenario."""
        h = _run(self.node_in, ((node_x - self.nm) / self.ns).astype(np.float32))
        g = _run(self.glob_in, ((glob_x - self.gm) / self.gs).astype(np.float32))
        gn = np.broadcast_to(g, h.shape)
        for (wm, bm), upd, (gam, bet, eps) in zip(self.msg, self.upd, self.norm):
            m = self.adj @ (h @ wm + bm)
            z = h + _run(upd, np.concatenate([h, m, gn], axis=1))
            mu = z.mean(axis=1, keepdims=True)
            var = z.var(axis=1, keepdims=True)
            h = (z - mu) / np.sqrt(var + eps) * gam + bet
        node = _run(self.node_out, np.concatenate([h, gn], axis=1))
        glob = _run(self.glob_out, np.concatenate([h.mean(axis=0), h.max(axis=0), g]))
        return node, glob

    def predict_one(self, node_x, glob_x):
        """Same keys and scaling as train.predict_arrays, for a single scenario (no batch axis)."""
        node, glob = self.forward(node_x, glob_x)
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        return dict(damage=sig(node[:, 0]), detach_prob=sig(node[:, 1]),
                    reg=glob[:N_GLOBAL_REG] * self.reg_std + self.reg_mean, flags_prob=sig(glob[N_GLOBAL_REG:]))


class OnnxGraphNet:
    """The same network run by ONNX Runtime (optional dependency: pip install onnx onnxruntime).

    The model is exported to ONNX in memory once; a single call then costs about 0.15 ms on one
    CPU thread, against about 0.45 ms for the NumPy version and 1 ms for PyTorch."""

    def __init__(self, bundle, threads=1):
        import io

        import onnxruntime as ort
        import torch

        class _Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, node_x, glob_x):
                o = self.m(node_x, glob_x)
                return o["damage"], o["detach"], o["reg"], o["flags"]

        nx = torch.zeros(1, 28, bundle.n_node_feat)
        gx = torch.zeros(1, bundle.n_glob_feat)
        buf = io.BytesIO()
        torch.onnx.export(_Wrap(bundle.model).eval(), (nx, gx), buf, input_names=["node_x", "glob_x"],
                          output_names=["damage", "detach", "reg", "flags"], dynamo=False)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(buf.getvalue(), so, providers=["CPUExecutionProvider"])
        nrm = bundle.norm
        self.nm, self.ns = nrm.node_mean.astype(np.float32), nrm.node_std.astype(np.float32)
        self.gm, self.gs = nrm.glob_mean.astype(np.float32), nrm.glob_std.astype(np.float32)
        self.reg_mean, self.reg_std = bundle.reg_mean, bundle.reg_std

    def predict_one(self, node_x, glob_x):
        nx = ((node_x - self.nm) / self.ns).astype(np.float32)[None]
        gx = ((glob_x - self.gm) / self.gs).astype(np.float32)[None]
        damage, detach, reg, flags = self.session.run(None, {"node_x": nx, "glob_x": gx})
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        return dict(damage=sig(damage[0]), detach_prob=sig(detach[0]),
                    reg=reg[0] * self.reg_std + self.reg_mean, flags_prob=sig(flags[0]))


def fast_runner(bundle):
    """ONNX Runtime if it is installed, otherwise NumPy. None for models other than the graph net."""
    if not isinstance(bundle.model, GraphNet):
        return None
    try:
        return OnnxGraphNet(bundle)
    except ImportError:
        return FastGraphNet(bundle)
