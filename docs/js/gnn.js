// The trained graph network, run in the browser (crashdamage/models.py GraphNet, train.py predict_arrays,
// export.py Predictor._from_model). Weights come from data/model.bin, written by scripts/export_web_data.py.
import { T } from './tables.js';
import { gelu, sigmoid } from './mathx.js';
import { scenarioInputs } from './features.js';
import { contactGeometry, impactEnergy, levelOf, effectHints } from './physics.js';

export const M = {}; // the loaded model

export function loadModel(manifest, buffer) {
  const t = {};
  for (const e of manifest.tensors) {
    t[e.name] = new Float32Array(buffer, e.offset, e.shape.reduce((a, b) => a * b, 1));
  }
  Object.assign(M, { manifest, cfg: manifest.config, nNode: manifest.n_node_feat, nGlob: manifest.n_glob_feat, t });
  return M;
}

// y = x W^T + b for `rows` rows
function linear(x, rows, inDim, name, outDim) {
  const W = M.t[`${name}.weight`], b = M.t[`${name}.bias`];
  const out = new Float64Array(rows * outDim);
  for (let r = 0; r < rows; r++) {
    const xo = r * inDim;
    for (let o = 0; o < outDim; o++) {
      let acc = b[o];
      const wo = o * inDim;
      for (let k = 0; k < inDim; k++) acc += W[wo + k] * x[xo + k];
      out[r * outDim + o] = acc;
    }
  }
  return out;
}

// Linear, GELU, Linear (the small networks of the model have no activation after the last layer)
function mlp(x, rows, inDim, name, hidden, outDim) {
  const a = linear(x, rows, inDim, `${name}.0`, hidden);
  for (let i = 0; i < a.length; i++) a[i] = gelu(a[i]);
  return linear(a, rows, hidden, `${name}.2`, outDim);
}

function layerNorm(x, rows, dim, name) {
  const w = M.t[`${name}.weight`], b = M.t[`${name}.bias`];
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let k = 0; k < dim; k++) mean += x[r * dim + k];
    mean /= dim;
    let v = 0;
    for (let k = 0; k < dim; k++) v += (x[r * dim + k] - mean) ** 2;
    v /= dim;
    const inv = 1 / Math.sqrt(v + 1e-5);
    for (let k = 0; k < dim; k++) x[r * dim + k] = (x[r * dim + k] - mean) * inv * w[k] + b[k];
  }
  return x;
}

// node_x: (parts x nNode) and glob_x: (nGlob), both already normalised. Returns the raw network outputs.
export function forward(nodeX, globX) {
  const N = T.nParts, H = M.cfg.hidden, adj = M.t.adj;
  const g = mlp(globX, 1, M.nGlob, 'glob_in', H, H);
  let h = mlp(nodeX, N, M.nNode, 'node_in', H, H);
  const cat3 = new Float64Array(N * 3 * H);
  for (let k = 0; k < M.cfg.layers; k++) {
    const mm = linear(h, N, H, `msg.${k}`, H);
    for (let i = 0; i < N; i++) {
      const o = i * 3 * H;
      for (let c = 0; c < H; c++) { cat3[o + c] = h[i * H + c]; cat3[o + H + c] = 0; cat3[o + 2 * H + c] = g[c]; }
      for (let j = 0; j < N; j++) {
        const a = adj[i * N + j];
        if (a === 0) continue;
        for (let c = 0; c < H; c++) cat3[o + H + c] += a * mm[j * H + c];
      }
    }
    const u = mlp(cat3, N, 3 * H, `upd.${k}`, H, H);
    for (let i = 0; i < h.length; i++) h[i] += u[i];
    h = layerNorm(h, N, H, `norm.${k}`);
  }
  const cat2 = new Float64Array(N * 2 * H);
  for (let i = 0; i < N; i++) {
    for (let c = 0; c < H; c++) { cat2[i * 2 * H + c] = h[i * H + c]; cat2[i * 2 * H + H + c] = g[c]; }
  }
  const node = mlp(cat2, N, 2 * H, 'node_out', H, 2);
  const pooled = new Float64Array(3 * H);
  for (let c = 0; c < H; c++) {
    let s = 0, mx = -Infinity;
    for (let i = 0; i < N; i++) { s += h[i * H + c]; mx = Math.max(mx, h[i * H + c]); }
    pooled[c] = s / N; pooled[H + c] = mx; pooled[2 * H + c] = g[c];
  }
  const glob = mlp(pooled, 1, 3 * H, 'glob_out', H, 7);
  return { node, glob };
}

// Raw outputs to numbers (predict_arrays): sigmoid for damage, detach and flags; regression outputs un-normalised.
export function predictArrays(node, glob) {
  const N = T.nParts, ns = M.t['norm.node_std'], nm = M.t['norm.node_mean'], gs = M.t['norm.glob_std'],
    gm = M.t['norm.glob_mean'];
  const nodeN = new Float32Array(node.length), globN = new Float32Array(glob.length);
  for (let i = 0; i < N; i++) for (let f = 0; f < M.nNode; f++) nodeN[i * M.nNode + f] = (node[i * M.nNode + f] - nm[f]) / ns[f];
  for (let f = 0; f < M.nGlob; f++) globN[f] = (glob[f] - gm[f]) / gs[f];
  const out = forward(nodeN, globN);
  const damage = new Float64Array(N), detachProb = new Float64Array(N);
  for (let i = 0; i < N; i++) { damage[i] = sigmoid(out.node[i * 2]); detachProb[i] = sigmoid(out.node[i * 2 + 1]); }
  const rm = M.t.reg_mean, rs = M.t.reg_std;
  const reg = [0, 1, 2, 3].map((k) => out.glob[k] * rs[k] + rm[k]);
  const flags = [4, 5, 6].map((k) => sigmoid(out.glob[k]));
  return { damage, detachProb, reg, flags };
}

// Damage report from the network. Same result shape as simulate(); the numbers the network does not predict are null.
export function predictModel(sc) {
  const C = T.constants, n = T.nParts;
  const geo = contactGeometry(sc);
  const inp = scenarioInputs(sc, geo);
  const out = predictArrays(inp.node, inp.glob);
  const damage = new Float64Array(n), level = new Array(n), detached = new Array(n), direct = new Array(n);
  for (let i = 0; i < n; i++) {
    let d = Math.min(Math.max(out.damage[i], 0), 1);
    if (d < 0.01) d = 0.0;
    damage[i] = d;
    level[i] = levelOf(d);
    detached[i] = out.detachProb[i] >= 0.5;
    direct[i] = geo.overlap[i] > C.MIN_OVERLAP;
  }
  const [crush, intrusion, peakMn, obstacleDamageRaw] = out.reg;
  const obstacleDamage = Math.min(Math.max(obstacleDamageRaw, 0), 1);
  const { e_diss: eDiss, dv, v_obs: vObs } = impactEnergy(sc, geo);
  const res = {
    engine: 'model', damage, level, detached, direct, crush_depth: Math.max(crush, 0), intrusion: Math.max(intrusion, 0),
    peak_force_kn: Math.max(peakMn, 0) * 1000.0, e_diss_kj: eDiss / 1e3, e_car_kj: null, e_obstacle_kj: null,
    unabsorbed_kj: null, obstacle_damage: obstacleDamage, obstacle_level: levelOf(obstacleDamage), obstacle_depth: null,
    breakthrough: null, airbag_front: out.flags[0] >= 0.5, airbag_side: out.flags[1] >= 0.5, delta_v: dv,
    obstacle_speed_after: vObs, through: out.flags[2] >= 0.5, capacity_kj: null, geo,
    raw: { detach_prob: out.detachProb, flags_prob: out.flags },
  };
  res.effects = effectHints(sc, res);
  return res;
}
