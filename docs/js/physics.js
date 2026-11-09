// JavaScript copy of the physics layer (crashdamage/physics.py, obstacles.py, vehicle.py).
// The code follows the Python line by line so that both give the same numbers; tests/web/check_parity.mjs checks that.
// It is a simplified energy model for visual plausibility, not a validated crash-engineering model.
import { T } from './tables.js';
import { sumNp, radians, pyRound } from './mathx.js';

const OBSTACLE_FIELDS = ['name', 'material', 'shape', 'form', 'width', 'depth', 'height', 'wall_mm', 'perimeter',
  'mass_kg', 'after', 'stroke', 'z_base'];

// ---- obstacles and scenarios ------------------------------------------------------------------------------

export function defaultObstacle(name) {
  return { ...T.obstacle_presets[name].default };
}

// Obstacle from the JSON the game engine sends (see obstacle_from_dict in obstacles.py).
// 0 and "" mean "keep the preset value" (except z_base); "fixed": true removes the mass.
export function obstacleFromDict(input) {
  const d = { ...input };
  const name = d.name || d.preset || 'concrete_wall';
  const baseName = d.base || name;
  if (!(baseName in T.obstacle_presets)) {
    throw new Error(`unknown obstacle '${baseName}'; presets: ${Object.keys(T.obstacle_presets).join(', ')}`);
  }
  const over = {};
  for (const [key, value] of Object.entries(d)) {
    if (!OBSTACLE_FIELDS.includes(key) || key === 'name' || value === null || value === undefined || value === '') continue;
    if (typeof value === 'number' && value === 0 && key !== 'z_base') continue;
    over[key] = value;
  }
  if ('material' in over && !(over.material in T.materials)) throw new Error(`unknown material '${over.material}'`);
  for (const [key, allowed] of [['shape', T.shapes], ['form', T.forms], ['after', T.after_modes]]) {
    if (key in over && !allowed.includes(over[key])) throw new Error(`${key} must be one of ${allowed.join(', ')}`);
  }
  const ob = { ...defaultObstacle(baseName), name, ...over };
  if (d.fixed || d.fixed_in_place) ob.mass_kg = null;
  return ob;
}

const pairs = (x, valueKey) => {
  if (Array.isArray(x)) {
    const out = {};
    for (const item of x) out[item.part] = item[valueKey];
    return out;
  }
  return { ...(x || {}) };
};

export function yawInertia(vehicle) {
  const c = T.constants;
  return vehicle.mass_kg * (c.CAR_LENGTH * c.CAR_LENGTH + c.CAR_WIDTH * c.CAR_WIDTH) / 12.0;
}

// Scenario from the request JSON (see scenario_from_dict in physics.py).
export function scenarioFromDict(d) {
  const vd = d.vehicle || {};
  const mass = Number(vd.mass_kg ?? 1100.0) || 1100.0;
  const scale = Number(vd.thickness_scale ?? 1.0) || 1.0;
  const mats = pairs(vd.materials, 'material');
  const thick = pairs(vd.thickness_mm, 'mm');
  const parts = T.parts.map((p) => {
    const m = p.name in mats ? mats[p.name] : p.material;
    if (!(m in T.materials)) throw new Error(`unknown material '${m}' for part ${p.name}`);
    const t = Number(p.name in thick ? thick[p.name] : p.thickness_mm * scale);
    return { ...p, material: m, thickness_mm: t };
  });
  const obstacle = obstacleFromDict(d.obstacle || { name: 'concrete_wall' });
  const speed = Number(d.speed_ms || 0.0) > 0 ? Number(d.speed_ms) : Number(d.speed_kmh ?? 50.0) / 3.6;
  const sc = {
    vehicle: { mass_kg: mass, parts },
    obstacle,
    face: d.face ?? 'front',
    offset: Number(d.offset ?? 0.0),
    angle_deg: Number(d.angle_deg ?? 0.0),
    speed,
  };
  if (!T.faces.includes(sc.face)) throw new Error(`face must be one of ${T.faces.join(', ')}`);
  if (!(sc.offset >= -1 && sc.offset <= 1)) throw new Error('offset must be between -1 and 1');
  if (!(sc.speed >= 0)) throw new Error('speed must be >= 0');
  return sc;
}

// ---- contact geometry and forces --------------------------------------------------------------------------

const axes = (face) => {
  const ax = face === 'front' || face === 'rear' ? 0 : 1;
  const sign = face === 'front' || face === 'left' ? 1.0 : -1.0;
  return [ax, 1 - ax, sign];
};

export function levelOf(d) {
  let k = 0;
  for (const e of T.level_edges) if (e <= d) k++;
  return k;
}

// Everything that follows from the layout alone (no energy involved).
export function contactGeometry(sc) {
  const ob = sc.obstacle, face = sc.face, parts = sc.vehicle.parts;
  const H = T.constants.CAR_HEIGHT;
  const [ax, along, sign] = axes(face);
  const half = T.face_half_span[face];
  const width = Math.min(ob.width, 2 * half);
  const c = sc.offset * (half - width / 2);
  const lo = c - width / 2, hi = c + width / 2;
  let z0 = Math.max(0.0, ob.z_base), z1 = Math.min(H, ob.z_base + ob.height);
  if (z1 <= z0) { // the object passes over the car; it touches the roof line
    z0 = H - 0.05;
    z1 = H;
  }
  const n = parts.length;
  const overlap = new Float64Array(n), depthStart = new Float64Array(n), stroke = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = parts[i];
    const qMax = sign * p.center[ax] + p.size[ax] / 2;
    depthStart[i] = Math.max(0.0, T.face_plane[face] - qMax);
    stroke[i] = p.size[ax] * p.crush_eff;
    const a0 = p.center[along] - p.size[along] / 2, a1 = p.center[along] + p.size[along] / 2;
    const ovA = Math.max(Math.min(hi, a1) - Math.max(lo, a0), 0) / p.size[along];
    const zp0 = p.center[2] - p.size[2] / 2, zp1 = p.center[2] + p.size[2] / 2;
    const ovZ = Math.max(Math.min(z1, zp1) - Math.max(z0, zp0), 0) / p.size[2];
    overlap[i] = Math.min(Math.max(ovA * ovZ, 0), 1);
  }
  // contact point (plan view and height) and the distance of every part to it
  const point = [0, 0, 0];
  point[ax] = sign * T.face_plane[face];
  point[along] = c;
  point[2] = 0.5 * (z0 + z1);
  const dist = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const ctr = parts[i].center;
    const dx = ctr[0] - point[0], dy = ctr[1] - point[1], dz = ctr[2] - point[2];
    dist[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
  }
  return { ax, along, sign, width, centre: c, lo, hi, z0, z1, overlap, depth_start: depthStart, stroke, dist,
    arm: Math.abs(c), point };
}

// Force of every part in every depth segment.
export function carForces(sc, geo) {
  const C = T.constants, parts = sc.vehicle.parts, n = parts.length, ax = geo.ax;
  const cutSheet = T.shape_cut[sc.obstacle.shape];
  const fPart = new Float64Array(n), engaged = new Array(n), start = geo.depth_start;
  for (let i = 0; i < n; i++) {
    const p = parts[i];
    const sigma = T.materials[p.material].strength_mpa * 1e6;
    const t = p.thickness_mm * 1e-3;
    const cut = t * 1e3 < C.SHEET_MM ? cutSheet : 1.0;
    const fFull = C.FOLD * sigma * t * p.section[ax] * cut;
    engaged[i] = geo.overlap[i] > C.MIN_OVERLAP;
    fPart[i] = engaged[i] ? fFull * geo.overlap[i] : 0.0;
  }
  const end = Float64Array.from(start, (s, i) => s + geo.stroke[i]);
  const maxDepth = T.extent[sc.face]; // nothing can be crushed deeper than the car is long (or wide)
  const pts = [0.0];
  for (let i = 0; i < n; i++) if (engaged[i]) pts.push(start[i]);
  for (let i = 0; i < n; i++) if (engaged[i]) pts.push(end[i]);
  pts.push(maxDepth);
  const sorted = pts.map((x) => Math.min(Math.max(x, 0.0), maxDepth)).sort((a, b) => a - b);
  const b = sorted.filter((x, i) => i === 0 || x !== sorted[i - 1]); // numpy.unique
  const packed = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    packed[i] = parts[i].crush_eff < C.BLOCK_EFF ? Math.min(C.DENS * fPart[i], C.F_BLOCK_CAP) : 0.0; // soft parts are used up, blocks stop the car
  }
  const f = [];
  for (let k = 0; k < b.length - 1; k++) {
    const mid = 0.5 * (b[k] + b[k + 1]);
    const row = new Float64Array(n);
    for (let i = 0; i < n; i++) row[i] = mid < start[i] ? 0.0 : (mid < end[i] ? fPart[i] : packed[i]);
    f.push(row);
  }
  return { b, f, fPart, engaged };
}

// Energy (kJ) each loaded part can take by crushing along its stroke, before it is used up.
export function partCapacityKj(sc, geo) {
  const { fPart, engaged } = carForces(sc, geo);
  return Float64Array.from(fPart, (f, i) => (engaged[i] ? f * geo.stroke[i] : 0.0) / 1e3);
}

// Force (N) at which the obstacle starts to give way (Obstacle.plateau_force).
export function plateauForce(ob, contactWidth, contactHeight = 0.6) {
  const C = T.constants;
  const sigma = T.materials[ob.material].strength_mpa * 1e6;
  const w = Math.min(ob.width, contactWidth);
  if (ob.form === 'solid') return C.SOLID_K * sigma * w * ob.depth;
  if (ob.form === 'thin_wall') return C.FOLD * sigma * ob.wall_mm * 1e-3 * ob.perimeter;
  if (ob.form === 'slender') return C.SLENDER_K * sigma * w * ob.depth;
  if (ob.form === 'foam') return sigma * w * Math.min(ob.height, contactHeight);
  throw new Error(`unknown form ${ob.form}`);
}

function obstacleCurve(sc, geo) {
  const ob = sc.obstacle, C = T.constants;
  const fO = plateauForce(ob, geo.width, geo.z1 - geo.z0);
  const after = ob.after === 'rigid' ? C.RIGID_FORCE : 0.0;
  return { b: [0.0, Math.max(ob.stroke, 1e-6), C.BIG_DEPTH], f: [fO, after], fO };
}

// March the two bodies into each other. The body with the lower current resistance gives way.
function solve(bc, fc, bo, foSeg, eDiss) {
  const nSeg = fc.length, n = fc[0].length;
  const fcTot = fc.map((row) => sumNp(row));
  const ahead = new Array(nSeg);
  let any = false;
  for (let k = nSeg - 1; k >= 0; k--) {
    any = any || fcTot[k] > 0;
    ahead[k] = any;
  }
  const eP = new Float64Array(n), fMax = new Float64Array(n);
  let kc = 0, ko = 0, dc = 0.0, dO = 0.0;
  let eLeft = eDiss, eObs = 0.0, peak = 0.0;
  let breakthrough = false, through = false;
  for (let iter = 0; iter < 10000; iter++) {
    if (eLeft <= 1e-6) break;
    const fCar = fcTot[kc], fObs = foSeg[ko];
    if (fObs <= 0.0) {
      breakthrough = true;
      break;
    }
    if (fCar <= fObs) {
      if (fCar <= 0 && !ahead[kc]) { // nothing left in the line of the obstacle: it goes through
        through = true;
        break;
      }
      const length = bc[kc + 1] - dc;
      const adv = fCar <= 0 ? length : Math.min(length, eLeft / fCar);
      for (let i = 0; i < n; i++) {
        eP[i] += fc[kc][i] * adv;
        fMax[i] = Math.max(fMax[i], fc[kc][i]);
      }
      peak = Math.max(peak, fCar);
      eLeft -= fCar * adv;
      dc += adv;
      if (adv >= length - 1e-12 && eLeft > 1e-6) {
        if (kc + 2 < bc.length) {
          kc += 1;
        } else { // reached the far side of the car
          through = true;
          break;
        }
      }
    } else {
      const length = bo[ko + 1] - dO;
      const adv = Math.min(length, eLeft / fObs);
      eObs += fObs * adv;
      eLeft -= fObs * adv;
      dO += adv;
      for (let i = 0; i < n; i++) fMax[i] = Math.max(fMax[i], fc[kc][i] * (fObs / fCar));
      peak = Math.max(peak, fObs);
      if (adv >= length - 1e-12 && ko + 2 < bo.length) ko += 1;
    }
  }
  return { e_p: eP, f_max: fMax, dc, do: dO, e_obs: eObs, peak, e_left: Math.max(eLeft, 0.0), breakthrough, through };
}

// Energy (J) that is dissipated, and the velocity change of the car.
export function impactEnergy(sc, geo) {
  const C = T.constants;
  const mC = sc.vehicle.mass_kg, ob = sc.obstacle;
  const fixed = ob.mass_kg === null || ob.mass_kg === undefined;
  const invEffCar = 1.0 / mC + (geo.arm * geo.arm) / yawInertia(sc.vehicle);
  const invMu = invEffCar + (fixed ? 0.0 : 1.0 / ob.mass_kg);
  const mu = 1.0 / invMu;
  const vN = sc.speed * Math.cos(radians(sc.angle_deg));
  const eDiss = 0.5 * mu * (vN * vN) * (1 - C.RESTITUTION * C.RESTITUTION);
  const impulse = (1 + C.RESTITUTION) * mu * vN;
  const dv = impulse / mC;
  const vObs = fixed ? 0.0 : impulse / ob.mass_kg;
  return { e_diss: eDiss, dv, v_obs: vObs, v_n: vN };
}

// ---- the whole simulation ---------------------------------------------------------------------------------

export function simulate(sc) {
  const C = T.constants, parts = sc.vehicle.parts, n = parts.length;
  const geo = contactGeometry(sc);
  const { e_diss: eDiss, dv, v_obs: vObs } = impactEnergy(sc, geo);
  const { b: bc, f: fc, fPart, engaged } = carForces(sc, geo);
  const { b: bo, f: foSeg, fO } = obstacleCurve(sc, geo);
  const sol = solve(bc, fc, bo, foSeg, eDiss);

  const capE = Float64Array.from(fPart, (f, i) => (engaged[i] ? f * geo.stroke[i] : 0.0)); // J
  const direct = new Array(n), dDir = new Float64Array(n), fDir = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const plastic = capE[i] > 0 ? sol.e_p[i] / capE[i] : 0.0;
    const elastic = C.ELASTIC_FLOOR * (fPart[i] > 0 ? Math.min(sol.f_max[i], fPart[i]) / fPart[i] : 0.0);
    direct[i] = engaged[i] && sol.f_max[i] > 0;
    dDir[i] = direct[i] ? Math.min(Math.max(Math.max(plastic, elastic), 0), 1) : 0.0;
    fDir[i] = direct[i] ? sol.f_max[i] : 0.0;
  }

  // Force travels through the part graph. A part cannot pass on more than it can carry,
  // and a part that only receives force through its neighbours is damaged in proportion.
  const fYield = new Float64Array(n), g = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = parts[i];
    const sigma = T.materials[p.material].strength_mpa * 1e6;
    const meanSection = (p.section[0] + p.section[1] + p.section[2]) / 3;
    fYield[i] = C.FOLD * sigma * (p.thickness_mm * 1e-3) * meanSection;
    g[i] = Math.min(Math.max(T.materials[p.material].fracture_strain / 0.15, 0.2), 1.0); // ductility_factor
  }
  const a = T.adj;
  const fIn = new Float64Array(n), dIn = new Float64Array(n);
  for (let hop = 0; hop < C.PROPAGATION_HOPS; hop++) {
    const decay = hop === 0 ? 1.0 : C.HOP_DECAY;
    const carried = new Float64Array(n), nbSrc = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      carried[i] = Math.min(Math.max(fDir[i], fIn[i]), fYield[i]); // a part passes on no more than it carries
      nbSrc[i] = Math.max(dDir[i], dIn[i]);
    }
    const nbCap = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      let mf = -Infinity, md = -Infinity;
      for (let i = 0; i < n; i++) {
        mf = Math.max(mf, a[i * n + j] * carried[i]);
        md = Math.max(md, a[i * n + j] * nbSrc[i]);
      }
      fIn[j] = Math.max(fIn[j], mf * decay);
      nbCap[j] = md * decay; // never worse than the neighbour it hangs on
    }
    for (let j = 0; j < n; j++) {
      const dNew = Math.min(C.INDIRECT_GAIN * Math.min(1.0, fIn[j] / fYield[j]) / g[j], nbCap[j]);
      dIn[j] = Math.max(dIn[j], Math.min(dNew, 1.0));
    }
  }
  const damage = new Float64Array(n), level = new Array(n), detached = new Array(n);
  for (let i = 0; i < n; i++) {
    let d = Math.min(Math.max(Math.max(dDir[i], dIn[i]), 0), 1);
    if (d < 0.01) d = 0.0;
    damage[i] = d;
    const f = Math.max(fDir[i], fIn[i]);
    detached[i] = f >= parts[i].attach_kn * 1e3 && d >= C.DETACH_MIN_DAMAGE;
    level[i] = levelOf(d);
  }

  // obstacle
  const ob = sc.obstacle;
  const capO = fO * ob.stroke;
  let dO = capO > 0 ? Math.min(1.0, sol.e_obs / capO) : 0.0;
  if (sol.breakthrough && ob.after === 'break') dO = 1.0;

  const depth = sol.dc;
  const intrusion = Math.max(0.0, depth - T.cabin_depth[sc.face]);
  const faceFront = sc.face === 'front' && Math.abs(sc.angle_deg) <= 30;
  const side = sc.face === 'left' || sc.face === 'right';
  const res = {
    engine: 'physics', damage, level, detached, direct, crush_depth: depth, intrusion,
    peak_force_kn: sol.peak / 1e3, e_diss_kj: eDiss / 1e3, e_car_kj: sumNp(sol.e_p) / 1e3,
    e_obstacle_kj: sol.e_obs / 1e3, obstacle_damage: dO, obstacle_level: levelOf(dO), obstacle_depth: sol.do,
    breakthrough: sol.breakthrough, airbag_front: faceFront && dv >= C.AIRBAG_FRONT_DV,
    airbag_side: side && dv >= C.AIRBAG_SIDE_DV, delta_v: dv, obstacle_speed_after: vObs,
    through: sol.through, unabsorbed_kj: sol.e_left / 1e3, capacity_kj: Float64Array.from(capE, (x) => x / 1e3), geo,
  };
  res.effects = effectHints(sc, res);
  return res;
}

const METALS = ['mild_steel', 'high_strength_steel', 'ultra_high_strength_steel', 'aluminium'];

// Simple rules that tell the game engine which effects make sense.
export function effectHints(sc, res) {
  const parts = sc.vehicle.parts, n = parts.length;
  const hard = sc.obstacle.material === 'concrete' || sc.obstacle.material === 'steel';
  let metalHit = false, glassBreak = false;
  for (let i = 0; i < n; i++) {
    if (res.direct[i] && METALS.includes(parts[i].material) && res.level[i] >= 2) metalHit = true;
    if (parts[i].kind === 'glass' && res.level[i] >= 3) glassBreak = true;
  }
  const leak = ['radiator', 'engine'].some((name) => res.level[T.partIndex[name]] >= 3);
  const broken = res.detached.some(Boolean) || res.level.some((l) => l >= 4) || res.obstacle_damage >= 1.0 || res.through;
  return {
    sparks: hard && metalHit && res.e_diss_kj > 5.0,
    glass_shards: glassBreak,
    smoke: leak,
    fluid_leak: leak,
    debris: broken,
    dust: (sc.obstacle.material === 'concrete' || sc.obstacle.material === 'wood') && res.obstacle_level >= 2,
    airbag_front: res.airbag_front,
    airbag_side: res.airbag_side,
  };
}

// The damage report JSON (result_to_dict in physics.py). Fields the AI model does not predict are null in `res`
// and are left out, like Predictor does in Python.
export function resultToReport(sc, res) {
  const r = (x, k) => pyRound(x, k);
  const parts = T.partNames.map((id, i) => ({
    id, damage: r(res.damage[i], 4), level: res.level[i], level_name: T.level_names[res.level[i]],
    detached: Boolean(res.detached[i]), direct: Boolean(res.direct[i]),
  }));
  const obstacle = { name: sc.obstacle.name, material: sc.obstacle.material, damage: r(res.obstacle_damage, 4),
    level: res.obstacle_level, level_name: T.level_names[res.obstacle_level] };
  if (res.obstacle_depth !== null) {
    obstacle.deformation_m = r(res.obstacle_depth, 4);
    obstacle.broken_through = res.breakthrough;
  }
  obstacle.speed_after_ms = r(res.obstacle_speed_after, 3);
  const summary = { crush_depth_m: r(res.crush_depth, 4), cabin_intrusion_m: r(res.intrusion, 4),
    peak_force_kn: r(res.peak_force_kn, 2), energy_total_kj: r(res.e_diss_kj, 2) };
  if (res.e_car_kj !== null) {
    summary.energy_car_kj = r(res.e_car_kj, 2);
    summary.energy_obstacle_kj = r(res.e_obstacle_kj, 2);
    summary.energy_unabsorbed_kj = r(res.unabsorbed_kj, 2);
  }
  summary.went_through_car = res.through;
  summary.delta_v_ms = r(res.delta_v, 3);
  return {
    parts, obstacle, summary, effects: res.effects,
    impact: { face: sc.face, offset: sc.offset, angle_deg: sc.angle_deg, speed_ms: sc.speed,
      speed_kmh: r(sc.speed * 3.6, 1) },
    engine: res.engine,
  };
}
