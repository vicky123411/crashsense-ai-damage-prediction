// The static data of the model (parts, part graph, materials, presets, constants), loaded once from data/static.json.
// The same numbers live in crashdamage/vehicle.py, materials.py, obstacles.py and physics.py;
// scripts/export_web_data.py copies them here.

export const T = {};

export function loadTables(json) {
  Object.assign(T, json);
  T.parts = json.parts.map((p) => ({ ...p, attach_kn: p.attach_kn === null ? Infinity : p.attach_kn }));
  T.partNames = T.parts.map((p) => p.name);
  T.partIndex = Object.fromEntries(T.partNames.map((n, i) => [n, i]));
  T.nParts = T.parts.length;
  // symmetric matrix of connection weights, flat, row i * n + column j (weights are float32 values, as in Python)
  const n = T.nParts;
  T.adj = new Float64Array(n * n);
  for (const [i, j, w] of json.edges) {
    T.adj[i * n + j] = w;
    T.adj[j * n + i] = w;
  }
  return T;
}
