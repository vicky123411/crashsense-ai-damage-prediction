// Words for the page: part names, the plain-language summary, the "how wrecked" verdict, and the warning that the
// AI model has not seen inputs like these. Pure functions, no DOM.
import { T } from './tables.js';

export const PART_LABEL = {
  bumper_front: 'Front bumper', beam_front: 'Front crash beam', rail_front_l: 'Left front rail',
  rail_front_r: 'Right front rail', radiator: 'Radiator', engine: 'Engine', bonnet: 'Bonnet',
  fender_l: 'Left fender', fender_r: 'Right fender', windscreen: 'Windscreen', pillar_a_l: 'Left A-pillar',
  pillar_a_r: 'Right A-pillar', firewall: 'Firewall', door_l: 'Left door', door_r: 'Right door',
  pillar_b_l: 'Left B-pillar', pillar_b_r: 'Right B-pillar', sill_l: 'Left side sill', sill_r: 'Right side sill',
  floor_cabin: 'Floor', roof: 'Roof', rear_glass: 'Rear window', bumper_rear: 'Rear bumper',
  rear_structure: 'Rear structure', wheel_fl: 'Front left wheel', wheel_fr: 'Front right wheel',
  wheel_rl: 'Rear left wheel', wheel_rr: 'Rear right wheel',
};

export const MATERIAL_LABEL = {
  mild_steel: 'Mild steel', high_strength_steel: 'High-strength steel', ultra_high_strength_steel: 'Ultra-high-strength steel',
  aluminium: 'Aluminium', plastic: 'Plastic', glass: 'Glass', concrete: 'Concrete', steel: 'Steel', wood: 'Wood', foam: 'Foam',
};
export const SHAPE_LABEL = { flat: 'Flat', round: 'Round', sharp: 'Sharp edge' };
export const FORM_LABEL = { solid: 'Solid', thin_wall: 'Thin wall (hollow)', slender: 'Slender (bends or snaps)', foam: 'Foam' };
export const AFTER_LABEL = { rigid: 'Stays rigid', break: 'Breaks apart' };
export const FACE_LABEL = { front: 'Front', rear: 'Rear', left: 'Left side', right: 'Right side' };

// name in a sentence ("a steel pole") and in a menu
export const PRESET_NOUN = {
  concrete_wall: 'concrete wall', concrete_block: 'concrete block', steel_pole: 'steel pole', steel_edge: 'steel edge',
  wood_fence: 'wooden fence', wood_post: 'wooden post', foam_barrier: 'foam barrier', steel_crate: 'steel crate',
};
export const PRESET_MENU = {
  concrete_wall: 'Concrete wall (fixed)', concrete_block: 'Concrete block (can be pushed)', steel_pole: 'Steel pole (fixed)',
  steel_edge: 'Steel edge (sharp, fixed)', wood_fence: 'Wooden fence', wood_post: 'Wooden post',
  foam_barrier: 'Foam barrier', steel_crate: 'Steel crate (can be pushed)',
};

export const EFFECTS = [
  ['sparks', 'Sparks', 'Metal scrapes against a hard obstacle at high energy'],
  ['glass_shards', 'Glass shards', 'A window is crushed or destroyed'],
  ['smoke', 'Smoke', 'The radiator or engine is crushed'],
  ['fluid_leak', 'Fluid leak', 'The radiator or engine is crushed'],
  ['debris', 'Debris', 'A part tears off, a part is destroyed, the obstacle breaks, or it goes through the car'],
  ['dust', 'Dust', 'A concrete or wooden obstacle is dented or worse'],
  ['airbag_front', 'Front airbag', 'Frontal hit within 30 degrees and the car slows down by 5.5 m/s or more'],
  ['airbag_side', 'Side airbag', 'Side hit and the car changes speed by 4 m/s or more'],
];

export const SEVERITY = [
  { name: 'Barely touched', rule: 'No part is worse than scratched.' },
  { name: 'Light damage', rule: 'At least one part is dented, or three are scratched.' },
  { name: 'Moderate damage', rule: 'At least one part is crushed, or four are dented.' },
  { name: 'Heavy damage', rule: 'Two parts are destroyed, or five are crushed, or the crush reaches the passenger cell.' },
  { name: 'Wrecked', rule: 'Six parts are destroyed, or the passenger cell is pushed in by 0.5 m or more, or the obstacle goes through the car.' },
];

// how wrecked the car looks (a look, not a safety rating)
export function severity(res) {
  const count = (k) => res.level.filter((l) => l >= k).length;
  const destroyed = count(4), crushed = count(3), dented = count(2), scratched = count(1);
  if (res.through || destroyed >= 6 || res.intrusion >= 0.5) return 4;
  if (destroyed >= 2 || crushed >= 5 || res.intrusion > 0) return 3;
  if (crushed >= 1 || dented >= 4) return 2;
  if (dented >= 1 || scratched >= 3) return 1;
  return 0;
}

const list = (items) => (items.length <= 1 ? items.join('') : `${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}`);
const m2 = (x) => (Math.round(x * 100) / 100).toFixed(2);
const lower = (s) => s.charAt(0).toLowerCase() + s.slice(1);

export function placeOnFace(sc) {
  const o = sc.offset;
  if (Math.abs(o) < 0.15) return 'in the middle';
  if (sc.face === 'front' || sc.face === 'rear') return o > 0 ? "toward the car's left edge" : "toward the car's right edge";
  return o > 0 ? 'toward the front' : 'toward the rear';
}

export function obstacleNoun(ob) {
  const base = PRESET_NOUN[ob.name] || ob.name.replace(/_/g, ' ');
  return base;
}

// A few plain sentences about one result.
export function summaryText(sc, res) {
  const ob = sc.obstacle;
  const kmh = Math.round(sc.speed * 3.6);
  const where = sc.face === 'front' || sc.face === 'rear' ? `on its ${sc.face}` : `on its ${sc.face} side`;
  const sizeBit = ob.width < 2 ? ` (${m2(ob.width)} m wide)` : '';
  const out = [`A ${Math.round(sc.vehicle.mass_kg)} kg car hits a ${obstacleNoun(ob)}${sizeBit} at ${kmh} km/h ${where}, ${placeOnFace(sc)}.`];
  const cab = res.intrusion > 0.005
    ? `${m2(res.intrusion)} m of that is inside the passenger cell`
    : 'the passenger cell is not reached';
  out.push(`The car is crushed ${m2(res.crush_depth)} m deep; ${cab}.`);
  const names = T.partNames;
  const worst = names.map((n, i) => ({ n, d: res.damage[i], l: res.level[i] })).filter((x) => x.l >= 3)
    .sort((a, b) => b.d - a.d).slice(0, 3).map((x) => lower(PART_LABEL[x.n]));
  if (worst.length) out.push(`Worst hit: ${list(worst)}.`);
  const torn = names.filter((n, i) => res.detached[i]).map((n) => lower(PART_LABEL[n]));
  if (torn.length) out.push(`${torn.length > 4 ? `${torn.length} parts tear off, among them` : 'Torn off:'} ${list(torn.slice(0, 4))}.`);
  if (res.through) out.push('The obstacle goes through the whole car; read the damage as destroyed along the line of the hit.');
  else if (ob.mass_kg && res.obstacle_speed_after > 0.3) {
    out.push(`The ${obstacleNoun(ob)} is pushed away at ${Math.round(res.obstacle_speed_after * 3.6)} km/h.`);
  }
  return out.join(' ');
}

export function obstacleLine(sc, res) {
  const ob = sc.obstacle, lvl = T.level_names[res.obstacle_level];
  const bits = [`${lvl[0].toUpperCase()}${lvl.slice(1)}`];
  if (res.obstacle_depth !== null && res.obstacle_depth > 0.005) bits.push(`gives way ${m2(res.obstacle_depth)} m`);
  if (res.breakthrough) bits.push('breaks');
  if (ob.mass_kg && res.obstacle_speed_after > 0.3) bits.push(`pushed away at ${Math.round(res.obstacle_speed_after * 3.6)} km/h`);
  else if (ob.mass_kg) bits.push('hardly moves');
  else bits.push('fixed in the ground');
  return bits.join(', ');
}

// ---- inputs the AI model has not seen ------------------------------------------------------------------------

export function trainingWarnings(sc) {
  const tr = T.training, out = [];
  const kmh = sc.speed * 3.6;
  const lo = tr.speed_ms[0] * 3.6, hi = tr.speed_ms[1] * 3.6;
  if (kmh < lo - 0.05) out.push(`speed ${Math.round(kmh)} km/h is below the ${Math.round(lo)} km/h the model saw`);
  if (kmh > hi + 0.05) out.push(`speed ${Math.round(kmh)} km/h is above the ${Math.round(hi)} km/h the model saw`);
  const mass = sc.vehicle.mass_kg;
  if (mass < tr.mass_kg[0] || mass > tr.mass_kg[1]) out.push(`car mass ${Math.round(mass)} kg is outside ${tr.mass_kg[0]} to ${tr.mass_kg[1]} kg`);
  if (Math.abs(sc.angle_deg) > tr.angle_deg[1] + 1e-9) out.push(`angle ${Math.round(sc.angle_deg)}° is beyond ±40°`);
  const tLo = tr.thickness_scale[0] * tr.thickness_jitter[0], tHi = tr.thickness_scale[1] * tr.thickness_jitter[1];
  const badT = [], badM = [];
  sc.vehicle.parts.forEach((p, i) => {
    const ratio = p.thickness_mm / T.parts[i].thickness_mm;
    if (ratio < tLo - 1e-9 || ratio > tHi + 1e-9) badT.push(p.name);
    if (!T.parts[i].options.includes(p.material)) badM.push(p.name);
  });
  if (badT.length) out.push(`sheet thickness of ${badT.length === 1 ? lower(PART_LABEL[badT[0]]) : `${badT.length} parts`} is outside ${Math.round(tLo * 100)} to ${Math.round(tHi * 100)} percent of normal`);
  if (badM.length) out.push(`${badM.length === 1 ? `the material of ${lower(PART_LABEL[badM[0]])}` : `the material of ${badM.length} parts`} was not used in training`);
  out.push(...obstacleWarnings(sc.obstacle));
  return out;
}

function obstacleWarnings(ob) {
  const preset = T.obstacle_presets[ob.name];
  if (!preset) return ['this obstacle is not one of the eight the model saw'];
  const spec = preset.spec, out = [];
  const label = obstacleNoun(ob);
  const tol = 1e-9;
  const named = { width: 'width', depth: 'depth', height: 'height', wall_mm: 'wall thickness', perimeter: 'loaded perimeter', stroke: 'give-way distance' };
  const unit = { width: 'm', depth: 'm', height: 'm', wall_mm: 'mm', perimeter: 'm', stroke: 'm' };
  // in the training data some sizes followed from others (a pole's depth equals its width, and so on)
  const derive = { w: ob.width, '2w': 2 * ob.width, 'w+d': ob.width + ob.depth, '0.6d': 0.6 * ob.depth };
  for (const [key, text] of Object.entries(named)) {
    const s = spec[key];
    if (s === undefined) continue;
    const u = unit[key], v = +ob[key].toFixed(3);
    if (Array.isArray(s)) {
      if (ob[key] < s[0] - tol || ob[key] > s[1] + tol) out.push(`${label} ${text} ${v} ${u} is outside ${s[0]} to ${s[1]} ${u}`);
    } else if (typeof s === 'string') {
      const want = derive[s];
      if (Math.abs(ob[key] - want) > 0.25 * want + 0.005) out.push(`${label} ${text} ${v} ${u} is far from ${+want.toFixed(3)} ${u}, the value that went with its size in training`);
    } else if (Math.abs(ob[key] - s) > 1e-9) out.push(`${label} ${text} differs from the preset`);
  }
  const fixed = ob.mass_kg === null || ob.mass_kg === undefined;
  if (Array.isArray(spec.mass_kg)) {
    if (fixed) out.push(`a ${label} was always free to move in training`);
    else if (ob.mass_kg < spec.mass_kg[0] - tol || ob.mass_kg > spec.mass_kg[1] + tol) out.push(`${label} mass ${Math.round(ob.mass_kg)} kg is outside ${spec.mass_kg[0]} to ${spec.mass_kg[1]} kg`);
  } else if (!fixed) out.push(`a ${label} was always fixed in the ground in training`);
  for (const key of ['material', 'shape', 'form', 'after']) {
    if (ob[key] !== spec[key]) out.push(`${key === 'after' ? 'what happens after it fails' : key} "${ob[key]}" is not what a ${label} had in training`);
  }
  if (ob.z_base > 0.5 + tol) out.push(`height above the ground ${+ob.z_base.toFixed(2)} m is above the 0.5 m the model saw`);
  return out;
}
