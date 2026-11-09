// The numbers that go into the network (crashdamage/features.py, vehicle.py, obstacles.py, materials.py).
// Only things known before the crash is solved are used: materials, thicknesses, layout, hit position, speed, masses.
import { T } from './tables.js';
import { radians, sumNp } from './mathx.js';
import { contactGeometry, carForces, impactEnergy } from './physics.js';

const materialFeatures = (name) => {
  const m = T.materials[name];
  return [Math.log10(m.density), Math.log10(m.youngs_gpa), Math.log10(m.strength_mpa), Math.log10(m.fracture_strain),
    m.brittle ? 1.0 : 0.0];
};

// 24 numbers describing one part
function partStaticFeatures(p) {
  const kind = T.kinds.map((k) => (p.kind === k ? 1.0 : 0.0));
  const attach = Number.isFinite(p.attach_kn) ? Math.log10(p.attach_kn) : 3.0;
  return [...materialFeatures(p.material), p.thickness_mm, Math.log10(p.mass_kg), ...p.center, ...p.size,
    p.crush_eff, attach, ...p.section, ...kind];
}

// 22 numbers describing the obstacle
function obstacleFeatures(ob) {
  const shape = T.shapes.map((s) => (ob.shape === s ? 1.0 : 0.0));
  const form = T.forms.map((f) => (ob.form === f ? 1.0 : 0.0));
  const mass = ob.mass_kg ? Math.log10(ob.mass_kg) : 0.0;
  const fixed = ob.mass_kg === null || ob.mass_kg === undefined;
  return [...materialFeatures(ob.material), ...shape, ...form, ob.width, ob.depth, ob.height, ob.wall_mm,
    ob.perimeter, mass, fixed ? 1.0 : 0.0, ob.after === 'break' ? 1.0 : 0.0, ob.stroke, ob.z_base];
}

// Model inputs of one scenario: per-part array (28 x 31), global array (39), already including the physics features.
export function scenarioInputs(sc, geo = contactGeometry(sc)) {
  const C = T.constants, parts = sc.vehicle.parts, n = parts.length;
  const { e_diss: eDiss } = impactEnergy(sc, geo);
  const { fPart, engaged } = carForces(sc, geo);
  const cap = Float64Array.from(fPart, (f, i) => (engaged[i] ? f * geo.stroke[i] : 0.0) / 1e3); // kJ
  const eDissKj = eDiss / 1e3;
  const logE = Math.log10(Math.max(eDissKj, 1e-3));
  const logCapTotal = Math.log10(Math.max(sumNp(cap), 1e-3));

  const nodeLen = 24 + 5 + 2;
  const node = new Float32Array(n * nodeLen); // Float32Array rounds like numpy float32 arrays
  for (let i = 0; i < n; i++) {
    const row = [...partStaticFeatures(parts[i]), geo.overlap[i], geo.depth_start[i], geo.stroke[i], geo.dist[i],
      geo.overlap[i] > C.MIN_OVERLAP ? 1.0 : 0.0, Math.log10(cap[i] + 1e-3), Math.log10(eDissKj / (cap[i] + 1e-3) + 1e-3)];
    node.set(row, i * nodeLen);
  }
  const vN = sc.speed * Math.cos(radians(sc.angle_deg));
  const face = T.faces.map((f) => (sc.face === f ? 1.0 : 0.0));
  const glob = Float32Array.from([...face, sc.speed, vN, sc.angle_deg, sc.offset, geo.width, geo.centre, geo.z0, geo.z1,
    geo.arm, Math.log10(sc.vehicle.mass_kg), ...obstacleFeatures(sc.obstacle), logE, logCapTotal, logE - logCapTotal]);
  return { node, glob, nodeLen, geo };
}
