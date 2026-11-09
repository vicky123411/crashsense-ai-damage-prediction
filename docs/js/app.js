// The page: inputs on the left, results on the right. Everything runs in the browser.
import { loadTables, T } from './tables.js';
import { loadModel, predictModel } from './gnn.js';
import { scenarioFromDict, simulate, resultToReport, defaultObstacle } from './physics.js';
import { createScene, LEVEL_FILL, LEVEL_TEXT } from './draw.js';
import {
  PART_LABEL, MATERIAL_LABEL, SHAPE_LABEL, FORM_LABEL, AFTER_LABEL, FACE_LABEL, PRESET_MENU, EFFECTS, SEVERITY,
  severity, summaryText, obstacleLine, trainingWarnings, obstacleNoun,
} from './describe.js';
import { h, num, copyText, nextId, debounce } from './dom.js';

const EXAMPLE_ORDER = ['side_pole_29kmh', 'front_wall_56kmh', 'offset_front_pole_50kmh', 'wood_fence_50kmh',
  'foam_barrier_56kmh', 'free_concrete_block_40kmh', 'mild_vs_ultra_high_strength_front'];
const EXAMPLE_TITLE = {
  side_pole_29kmh: 'Side pole, 29 km/h', front_wall_56kmh: 'Wall, 56 km/h',
  offset_front_pole_50kmh: 'Offset pole, 50 km/h', wood_fence_50kmh: 'Fence, 50 km/h',
  foam_barrier_56kmh: 'Foam barrier, 56 km/h', free_concrete_block_40kmh: 'Loose block, 40 km/h',
  mild_vs_ultra_high_strength_front: 'Strong rails, 56 km/h',
};
const PART_GROUPS = [
  ['Front bumper', ['bumper_front']], ['Front crash beam', ['beam_front']], ['Front rails', ['rail_front_l', 'rail_front_r']],
  ['Radiator', ['radiator']], ['Engine', ['engine']], ['Bonnet', ['bonnet']], ['Fenders', ['fender_l', 'fender_r']],
  ['Windscreen', ['windscreen']], ['A-pillars', ['pillar_a_l', 'pillar_a_r']], ['Firewall', ['firewall']],
  ['Doors', ['door_l', 'door_r']], ['B-pillars', ['pillar_b_l', 'pillar_b_r']], ['Side sills', ['sill_l', 'sill_r']],
  ['Floor', ['floor_cabin']], ['Roof', ['roof']], ['Rear window', ['rear_glass']], ['Rear bumper', ['bumper_rear']],
  ['Rear structure', ['rear_structure']], ['Wheels', ['wheel_fl', 'wheel_fr', 'wheel_rl', 'wheel_rr']],
];
const OB_LIMITS = {
  width: [0.02, 10], depth: [0.01, 3], height: [0.1, 6], z_base: [0, 3], wall_mm: [0.1, 60], perimeter: [0.05, 10],
  stroke: [0.001, 2], mass_kg: [20, 20000],
};
const ENGINE_COLOR = { model: '#2457D6', physics: '#3B4650' };
const ENGINE_NAME = { model: 'AI model', physics: 'Physics layer' };

const $ = (sel, root = document) => root.querySelector(sel);
const state = {
  engine: 'model', drawEngine: 'model', showAll: false, example: null,
  speed_kmh: 50, face: 'front', offset: 0, angle_deg: 0, mass_kg: 1100, thickness_scale: 1,
  partMaterial: {}, partThickness: {}, ob: null,
};
let hasModel = false;
let scene = null;
let lastRun = null;
const bindings = []; // functions that copy the state into the form controls

// ---- state <-> request JSON -----------------------------------------------------------------------------------------

function resetCar() {
  for (const p of T.parts) { state.partMaterial[p.name] = p.material; state.partThickness[p.name] = null; }
  state.mass_kg = 1100;
  state.thickness_scale = 1;
}

const effThickness = (name) => state.partThickness[name] ?? T.parts[T.partIndex[name]].thickness_mm * state.thickness_scale;

function newObstacle(preset) {
  const d = defaultObstacle(preset);
  const free = d.mass_kg !== null;
  const massGuess = free ? d.mass_kg : 1000;
  return { preset, ...d, fixed: !free, mass_kg: massGuess };
}

function toRequest() {
  const ob = state.ob;
  const obstacle = {
    name: ob.preset, material: ob.material, shape: ob.shape, form: ob.form, width: ob.width, depth: ob.depth,
    height: ob.height, wall_mm: ob.wall_mm, perimeter: ob.perimeter, after: ob.after, stroke: ob.stroke, z_base: ob.z_base,
  };
  if (ob.fixed) obstacle.fixed = true;
  else obstacle.mass_kg = ob.mass_kg;
  const materials = T.parts.filter((p) => state.partMaterial[p.name] !== p.material)
    .map((p) => ({ part: p.name, material: state.partMaterial[p.name] }));
  const thickness = T.parts.filter((p) => state.partThickness[p.name] !== null)
    .map((p) => ({ part: p.name, mm: state.partThickness[p.name] }));
  const vehicle = { mass_kg: state.mass_kg };
  if (state.thickness_scale !== 1) vehicle.thickness_scale = state.thickness_scale;
  if (materials.length) vehicle.materials = materials;
  if (thickness.length) vehicle.thickness_mm = thickness;
  return { speed_kmh: state.speed_kmh, face: state.face, offset: state.offset, angle_deg: state.angle_deg, vehicle, obstacle };
}

function fromRequest(req) {
  const sc = scenarioFromDict(req); // throws if the request is not valid
  resetCar();
  const vd = req.vehicle || {};
  state.mass_kg = sc.vehicle.mass_kg;
  state.thickness_scale = Number(vd.thickness_scale ?? 1) || 1;
  const thick = Array.isArray(vd.thickness_mm) ? Object.fromEntries(vd.thickness_mm.map((x) => [x.part, x.mm])) : (vd.thickness_mm || {});
  sc.vehicle.parts.forEach((p) => {
    state.partMaterial[p.name] = p.material;
    state.partThickness[p.name] = p.name in thick ? Number(thick[p.name]) : null;
  });
  state.speed_kmh = Number(req.speed_ms || 0) > 0 ? Math.round(Number(req.speed_ms) * 36) / 10 : Number(req.speed_kmh ?? 50);
  state.face = sc.face;
  state.offset = sc.offset;
  state.angle_deg = sc.angle_deg;
  const o = req.obstacle || { name: 'concrete_wall' };
  const preset = o.base || o.name || o.preset || 'concrete_wall';
  const ob = sc.obstacle;
  state.ob = { preset, ...ob, name: preset, fixed: ob.mass_kg === null, mass_kg: ob.mass_kg === null ? newObstacle(preset).mass_kg : ob.mass_kg };
}

const b64 = {
  enc: (s) => btoa(unescape(encodeURIComponent(s))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''),
  dec: (s) => decodeURIComponent(escape(atob(s.replace(/-/g, '+').replace(/_/g, '/')))),
};
let lastHash = ''; // the hash this page wrote itself, so that its own writes are not mistaken for a pasted link
function writeHash(req) {
  try {
    lastHash = `#s=${b64.enc(JSON.stringify({ r: req, e: state.engine }))}`;
    history.replaceState(null, '', lastHash);
  } catch { /* the page still works without it */ }
}
function readHash() {
  try {
    const m = /^#s=(.+)$/.exec(location.hash);
    if (!m) return false;
    const { r, e } = JSON.parse(b64.dec(m[1]));
    fromRequest(r);
    if (['model', 'physics', 'compare'].includes(e)) state.engine = e;
    return true;
  } catch { return false; }
}

// ---- form -----------------------------------------------------------------------------------------------------------

function changed() {
  state.example = null;
  syncExampleChips();
  schedule();
}

function section(title, ...kids) {
  const id = nextId('sec');
  return h('fieldset', { class: 'group', 'aria-labelledby': id }, h('legend', { id, class: 'group-title' }, title), ...kids);
}

function rangeField({ label, unit, min, max, step, get, set, digits = 0, endLabels, hint }) {
  const id = nextId('r'), nid = nextId('n');
  const range = h('input', { type: 'range', id, min, max, step, 'aria-label': label });
  const number = h('input', { type: 'number', id: nid, min, max, step, inputmode: 'decimal', 'aria-label': `${label} (number)` });
  const ends = endLabels ? h('div', { class: 'ends' }, h('span', { text: '' }), h('span', { text: '' })) : null;
  const wrap = h('div', { class: 'field' },
    h('div', { class: 'field-head' }, h('label', { for: id, text: label }), h('span', { class: 'with-unit' }, number, h('span', { class: 'unit', text: unit }))),
    range, ends, hint ? h('p', { class: 'hint', text: hint }) : null);
  const push = (v) => { if (Number.isFinite(v)) { set(v); changed(); } };
  range.addEventListener('input', () => { push(parseFloat(range.value)); number.value = num(range.value, digits + 2); number.removeAttribute('aria-invalid'); });
  number.addEventListener('input', () => {
    const v = parseFloat(number.value);
    if (Number.isFinite(v) && v >= min && v <= max) { number.removeAttribute('aria-invalid'); range.value = v; push(v); } else number.setAttribute('aria-invalid', 'true');
  });
  number.addEventListener('change', () => { number.value = num(get(), digits + 2); number.removeAttribute('aria-invalid'); });
  bindings.push(() => {
    const v = get();
    range.value = v;
    if (document.activeElement !== number) { number.value = num(v, digits + 2); number.removeAttribute('aria-invalid'); }
  });
  wrap.setEnds = (a, b) => { ends.children[0].textContent = a; ends.children[1].textContent = b; };
  return wrap;
}

function numberField({ label, unit, limits, step, get, set, only }) {
  const id = nextId('n');
  const [min, max] = limits;
  const input = h('input', { type: 'number', id, min, max, step, inputmode: 'decimal' });
  const wrap = h('div', { class: 'field compact' }, h('label', { for: id, text: label }), h('span', { class: 'with-unit' }, input, h('span', { class: 'unit', text: unit })));
  input.addEventListener('input', () => {
    const v = parseFloat(input.value);
    if (Number.isFinite(v) && v >= min && v <= max) { input.removeAttribute('aria-invalid'); set(v); changed(); } else input.setAttribute('aria-invalid', 'true');
  });
  input.addEventListener('change', () => { input.value = num(get(), 4); input.removeAttribute('aria-invalid'); });
  bindings.push(() => {
    // while someone is typing in this box the text is left alone, so "1.50" does not turn into "1.5" under their fingers
    if (document.activeElement !== input) { input.value = num(get(), 4); input.removeAttribute('aria-invalid'); }
    if (only) wrap.hidden = !only();
  });
  return wrap;
}

function selectField({ label, options, get, set, compact = true, disabledIf }) {
  const id = nextId('s');
  const sel = h('select', { id }, options.map(([v, t]) => h('option', { value: v, text: t })));
  sel.addEventListener('change', () => { set(sel.value); changed(); syncForm(); });
  bindings.push(() => {
    const v = get();
    if (![...sel.options].some((o) => o.value === v)) sel.append(h('option', { value: v, text: v }));
    sel.value = v;
  });
  return h('div', { class: `field${compact ? ' compact' : ''}` }, h('label', { for: id, text: label }), sel);
}

function segmented({ label, options, get, set, name }) {
  const gid = nextId('g');
  const wrap = h('div', { class: 'field' }, h('span', { class: 'label', id: gid, text: label }));
  const group = h('div', { class: 'segmented', role: 'radiogroup', 'aria-labelledby': gid });
  const inputs = options.map(([v, t]) => {
    const id = nextId('o');
    const input = h('input', { type: 'radio', name, id, value: v });
    input.addEventListener('change', () => { if (input.checked) { set(v); changed(); syncForm(); } });
    group.append(h('span', { class: 'seg' }, input, h('label', { for: id, text: t })));
    return input;
  });
  bindings.push(() => inputs.forEach((i) => { i.checked = i.value === get(); }));
  wrap.append(group);
  return wrap;
}

function checkField({ label, get, set }) {
  const id = nextId('c');
  const input = h('input', { type: 'checkbox', id });
  input.addEventListener('change', () => { set(input.checked); changed(); syncForm(); });
  bindings.push(() => { input.checked = get(); });
  return h('div', { class: 'field check' }, input, h('label', { for: id, text: label }));
}

const OFFSET_ENDS = {
  front: ["car's right edge", "car's left edge"], rear: ["car's right edge", "car's left edge"],
  left: ['rear end', 'front end'], right: ['rear end', 'front end'],
};

function deriveFromSize(changedKey) {
  const ob = state.ob, spec = T.obstacle_presets[ob.preset].spec;
  if (changedKey === 'width') {
    if (spec.depth === 'w') ob.depth = ob.width;
    if (spec.perimeter === '2w') ob.perimeter = 2 * ob.width;
  }
  if (changedKey === 'width' || changedKey === 'depth') {
    if (spec.perimeter === 'w+d') ob.perimeter = ob.width + ob.depth;
    if (spec.stroke === '0.6d') ob.stroke = 0.6 * ob.depth;
  }
}

function buildForm() {
  const form = $('#inputs');
  form.textContent = '';
  bindings.length = 0;

  const offsetField = rangeField({
    label: 'Where along that side', unit: '', min: -1, max: 1, step: 0.05, digits: 2, endLabels: true,
    get: () => state.offset, set: (v) => { state.offset = v; },
  });
  const hit = section('The hit',
    rangeField({ label: 'Speed', unit: 'km/h', min: 10, max: 140, step: 1, get: () => state.speed_kmh, set: (v) => { state.speed_kmh = v; } }),
    segmented({ label: 'Side of the car that is hit', name: 'face', options: T.faces.map((f) => [f, FACE_LABEL[f]]), get: () => state.face, set: (v) => { state.face = v; } }),
    offsetField,
    rangeField({ label: 'Angle of approach', unit: '°', min: -60, max: 60, step: 1, hint: '0° means straight on.', get: () => state.angle_deg, set: (v) => { state.angle_deg = v; } }));
  bindings.push(() => offsetField.setEnds(...OFFSET_ENDS[state.face]));

  const obstacleBasic = [
    selectField({ label: 'Obstacle', compact: false, options: Object.keys(T.obstacle_presets).map((k) => [k, PRESET_MENU[k]]),
      get: () => state.ob.preset, set: (v) => { state.ob = newObstacle(v); } }),
    selectField({ label: 'Material', compact: false, options: T.obstacle_materials.map((m) => [m, MATERIAL_LABEL[m]]),
      get: () => state.ob.material, set: (v) => { state.ob.material = v; } }),
    h('div', { class: 'pair' },
      numberField({ label: 'Width', unit: 'm', limits: OB_LIMITS.width, step: 0.01, get: () => state.ob.width, set: (v) => { state.ob.width = v; deriveFromSize('width'); syncForm(); } }),
      numberField({ label: 'Height', unit: 'm', limits: OB_LIMITS.height, step: 0.05, get: () => state.ob.height, set: (v) => { state.ob.height = v; } })),
    checkField({ label: 'Fixed in the ground', get: () => state.ob.fixed, set: (v) => { state.ob.fixed = v; } }),
    numberField({ label: 'Mass', unit: 'kg', limits: OB_LIMITS.mass_kg, step: 10, get: () => state.ob.mass_kg, set: (v) => { state.ob.mass_kg = v; }, only: () => !state.ob.fixed }),
  ];
  const obstacleMore = h('details', { class: 'more' }, h('summary', { text: 'More obstacle details' }),
    h('div', { class: 'pair' },
      selectField({ label: 'Shape', options: T.shapes.map((s) => [s, SHAPE_LABEL[s]]), get: () => state.ob.shape, set: (v) => { state.ob.shape = v; } }),
      selectField({ label: 'Build', options: T.forms.map((f) => [f, FORM_LABEL[f]]), get: () => state.ob.form, set: (v) => {
        state.ob.form = v;
        if (v === 'thin_wall') {
          if (!(state.ob.wall_mm > 0)) state.ob.wall_mm = 5;
          if (!(state.ob.perimeter > 0)) state.ob.perimeter = state.ob.width + state.ob.depth;
        }
      } })),
    h('div', { class: 'pair' },
      numberField({ label: 'Depth', unit: 'm', limits: OB_LIMITS.depth, step: 0.01, get: () => state.ob.depth, set: (v) => { state.ob.depth = v; deriveFromSize('depth'); syncForm(); } }),
      numberField({ label: 'Lowest point above ground', unit: 'm', limits: OB_LIMITS.z_base, step: 0.05, get: () => state.ob.z_base, set: (v) => { state.ob.z_base = v; } })),
    h('div', { class: 'pair' },
      numberField({ label: 'Wall thickness', unit: 'mm', limits: OB_LIMITS.wall_mm, step: 0.5, get: () => state.ob.wall_mm, set: (v) => { state.ob.wall_mm = v; }, only: () => state.ob.form === 'thin_wall' }),
      numberField({ label: 'Loaded perimeter', unit: 'm', limits: OB_LIMITS.perimeter, step: 0.05, get: () => state.ob.perimeter, set: (v) => { state.ob.perimeter = v; }, only: () => state.ob.form === 'thin_wall' })),
    h('div', { class: 'pair' },
      numberField({ label: 'Gives way over', unit: 'm', limits: OB_LIMITS.stroke, step: 0.005, get: () => state.ob.stroke, set: (v) => { state.ob.stroke = v; } }),
      selectField({ label: 'Once it gives way', options: T.after_modes.map((a) => [a, AFTER_LABEL[a]]), get: () => state.ob.after, set: (v) => { state.ob.after = v; } })),
    h('p', { class: 'hint', text: 'Changing the width of a pole or crate also updates its depth and perimeter, as in the data the model learned from.' }));
  const obstacle = section('The obstacle', ...obstacleBasic, obstacleMore);

  const partRows = PART_GROUPS.map(([title, names]) => {
    const mid = nextId('pm'), tid = nextId('pt');
    const sel = h('select', { id: mid }, [...T.vehicle_materials.map((m) => h('option', { value: m, text: MATERIAL_LABEL[m] })), h('option', { value: '__mixed', text: 'Mixed', disabled: true })]);
    const inp = h('input', { type: 'number', id: tid, min: 0.2, max: 40, step: 0.1, inputmode: 'decimal', 'aria-label': `${title} thickness in mm` });
    sel.addEventListener('change', () => { names.forEach((n) => { state.partMaterial[n] = sel.value; }); changed(); syncForm(); });
    inp.addEventListener('input', () => {
      const v = parseFloat(inp.value);
      if (Number.isFinite(v) && v >= 0.2 && v <= 40) { inp.removeAttribute('aria-invalid'); names.forEach((n) => { state.partThickness[n] = v; }); changed(); } else inp.setAttribute('aria-invalid', 'true');
    });
    bindings.push(() => {
      const mats = new Set(names.map((n) => state.partMaterial[n]));
      sel.value = mats.size === 1 ? [...mats][0] : '__mixed';
      const th = new Set(names.map((n) => +effThickness(n).toFixed(3)));
      inp.value = th.size === 1 ? num([...th][0], 3) : '';
      inp.placeholder = th.size === 1 ? '' : 'mixed';
    });
    return h('tr', null, h('th', { scope: 'row' }, h('label', { for: mid, text: title })), h('td', null, sel), h('td', null, h('span', { class: 'with-unit' }, inp, h('span', { class: 'unit', text: 'mm' }))));
  });
  const car = section('The car',
    rangeField({ label: 'Mass', unit: 'kg', min: 700, max: 2500, step: 10, get: () => state.mass_kg, set: (v) => { state.mass_kg = v; } }),
    rangeField({ label: 'Sheet thickness, all parts', unit: '×', min: 0.7, max: 1.5, step: 0.05, digits: 2, get: () => state.thickness_scale, set: (v) => { state.thickness_scale = v; },
      hint: 'Parts you set by hand below keep their own thickness.' }),
    h('details', { class: 'more' }, h('summary', { text: 'Material and thickness of each part' }),
      h('table', { class: 'part-edit' }, h('thead', null, h('tr', null, h('th', { scope: 'col', text: 'Part' }), h('th', { scope: 'col', text: 'Material' }), h('th', { scope: 'col', text: 'Thickness' }))), h('tbody', null, partRows)),
      h('button', { type: 'button', class: 'plain', onclick: () => { const m = state.mass_kg; resetCar(); state.mass_kg = m; changed(); syncForm(); }, text: 'Reset all parts to the standard car' })));

  form.append(hit, obstacle, car);
  syncForm();
}

function syncForm() { bindings.forEach((f) => f()); }

function buildExamples() {
  const wrap = $('#examples-list');
  wrap.textContent = '';
  EXAMPLE_ORDER.forEach((id) => {
    const ex = T.examples.find((e) => e.id === id);
    if (!ex) return;
    const b = h('button', { type: 'button', class: 'chip', 'data-id': id, title: ex.comment, text: EXAMPLE_TITLE[id] });
    b.addEventListener('click', () => { loadExample(id); });
    wrap.append(h('li', null, b));
  });
}
function syncExampleChips() {
  document.querySelectorAll('#examples-list .chip').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.id === state.example)));
}
function loadExample(id) {
  const ex = T.examples.find((e) => e.id === id);
  fromRequest(JSON.parse(JSON.stringify(ex.request)));
  state.example = id;
  syncForm();
  syncExampleChips();
  schedule();
}

// ---- results --------------------------------------------------------------------------------------------------------

// screen readers hear the result once the inputs have stopped changing, not on every tick of a slider
const announce = debounce((text) => { $('#live').textContent = text; }, 900);

let pending = false;
function schedule() {
  if (pending) return;
  pending = true;
  requestAnimationFrame(() => { pending = false; run(); });
}

function run() {
  let req, sc;
  try {
    req = toRequest();
    sc = scenarioFromDict(req);
  } catch (e) {
    showProblem(e.message);
    return;
  }
  const out = {};
  try {
    out.physics = simulate(sc); // always: it is cheap, and only this engine reports the energy split
    if (hasModel && state.engine !== 'physics') out.model = predictModel(sc);
  } catch (e) {
    showProblem(e.message);
    return;
  }
  hideProblem();
  render(sc, out);
  lastRun.req = req;
  writeHash(req);
}

const fmtNum = (x, d) => Number(x).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

function render(sc, out) {
  const compare = Boolean(out.model) && state.engine === 'compare';
  const drawKey = !out.model ? 'physics' : (compare && state.drawEngine === 'physics' ? 'physics' : 'model');
  const main = out[drawKey], other = compare ? out[drawKey === 'model' ? 'physics' : 'model'] : null;
  lastRun = { sc, out, drawKey };

  // verdict
  const sev = severity(main);
  $('#v-word').textContent = SEVERITY[sev].name;
  $('#v-scale').dataset.level = sev;
  [...$('#v-scale').children].forEach((c, i) => c.classList.toggle('on', i <= sev));
  $('#v-rule').textContent = SEVERITY[sev].rule;
  $('#summary').textContent = summaryText(sc, main);
  $('#mini-word').textContent = SEVERITY[sev].name;
  $('#mini-sub').textContent = `Crush ${fmtNum(main.crush_depth, 2)} m`;
  announce(`${SEVERITY[sev].name}. ${summaryText(sc, main)}`);
  const diff = $('#v-diff');
  if (other) {
    const so = severity(other);
    const d = main.damage.reduce((s, x, i) => s + Math.abs(x - other.damage[i]), 0) / main.damage.length;
    const lv = main.level.reduce((s, x, i) => s + (x !== other.level[i] ? 1 : 0), 0);
    diff.hidden = false;
    diff.textContent = `${ENGINE_NAME[drawKey === 'model' ? 'physics' : 'model']} says: ${SEVERITY[so].name.toLowerCase()}. The two engines differ by ${d.toFixed(3)} per part on average (damage runs from 0 to 1), and put ${lv} of 28 parts in a different level.`;
  } else diff.hidden = true;

  // notice for inputs the AI model has not seen
  const notice = $('#notice');
  const warn = out.model ? trainingWarnings(sc) : [];
  if (warn.length) {
    notice.hidden = false;
    $('#notice-list').textContent = warn.slice(0, 4).join('; ') + (warn.length > 4 ? `; and ${warn.length - 4} more` : '');
  } else notice.hidden = true;

  // drawing
  const alt = `${summaryText(sc, main)}`;
  scene.update(sc, main, { label: ENGINE_NAME[drawKey], color: ENGINE_COLOR[drawKey], alt });
  document.querySelectorAll('#draw-switch input').forEach((i) => { i.checked = i.value === drawKey; });
  $('#draw-switch').hidden = !compare;

  // readouts
  const set = (key, text, other2) => {
    const cell = $(`[data-ro="${key}"]`);
    $('.v', cell).firstChild.nodeValue = text[0];
    $('.u', cell).textContent = text[1];
    const o = $('.o', cell);
    if (other2) { o.hidden = false; o.textContent = other2; } else o.hidden = true;
  };
  const fmtKey = (r, key) => {
    if (r[key] === null || r[key] === undefined) return null;
    const x = key === 'delta_v' ? r[key] * 3.6 : r[key];
    return fmtNum(x, key === 'crush_depth' || key === 'intrusion' ? 2 : 0);
  };
  for (const [key, unit] of [['crush_depth', 'm'], ['intrusion', 'm'], ['peak_force_kn', 'kN'], ['delta_v', 'km/h'], ['e_diss_kj', 'kJ']]) {
    set(key, [fmtKey(main, key), unit], other ? `${ENGINE_NAME[drawKey === 'model' ? 'physics' : 'model']}: ${fmtKey(other, key)} ${unit}` : '');
  }
  // the energy split is only reported by the physics layer
  const split = out.physics, cell = $('[data-ro="split"]');
  $('.v', cell).firstChild.nodeValue = `${fmtNum(split.e_car_kj, 0)} / ${fmtNum(split.e_obstacle_kj, 0)}`;
  $('.u', cell).textContent = 'kJ';
  const kept = split.unabsorbed_kj > 0.5 ? `. ${fmtNum(split.unabsorbed_kj, 0)} kJ stays as motion` : '';
  $('.c', cell).textContent = `Car / obstacle${kept}${drawKey === 'model' ? ', from the physics layer' : ''}`;
  $('.o', cell).hidden = true;

  // effects
  for (const [key] of EFFECTS) {
    const li = $(`[data-fx="${key}"]`);
    const on = Boolean(main.effects[key]);
    li.classList.toggle('on', on);
    $('.state', li).textContent = on ? 'yes' : 'no';
    const alt2 = other && Boolean(other.effects[key]) !== on;
    li.classList.toggle('differs', Boolean(alt2));
  }
  $('#ob-line').textContent = obstacleLine(sc, main);
  const noun = obstacleNoun(sc.obstacle);
  $('#ob-name').textContent = noun.charAt(0).toUpperCase() + noun.slice(1);
  $('#fx-note').hidden = !other;

  renderParts(main, other, drawKey);
  document.body.dataset.engine = state.engine;
  document.body.dataset.renders = String((Number(document.body.dataset.renders) || 0) + 1); // lets the browser tests wait for a redraw
}

let hovered = null, pinned = null;
const showOutline = () => scene.highlight(hovered ?? pinned);

function renderParts(main, other, drawKey) {
  const tbody = $('#parts tbody');
  const order = T.partNames.map((n, i) => i).sort((a, b) => main.damage[b] - main.damage[a] || a - b);
  const rows = state.showAll ? order : order.slice(0, 10);
  tbody.textContent = '';
  const bar = (r, i, color) => h('span', { class: 'track' }, h('i', { style: `width:${Math.max(r.damage[i] * 100, r.damage[i] > 0 ? 2 : 0).toFixed(1)}%;background:${color || LEVEL_FILL[r.level[i]]}` }));
  for (const i of rows) {
    const name = T.partNames[i];
    const mk = (r, color) => h('span', { class: 'dmg' }, bar(r, i, color), h('span', { class: 'val', text: r.damage[i].toFixed(2) }));
    const lvl = (r) => h('span', { class: 'lvl', style: `background:${LEVEL_FILL[r.level[i]]};color:${LEVEL_TEXT[r.level[i]]}`, text: T.level_names[r.level[i]] });
    const notes = [];
    if (main.detached[i]) notes.push('torn off');
    if (main.direct[i]) notes.push('in the hit');
    if (other && other.detached[i] !== main.detached[i]) notes.push(other.detached[i] ? `${ENGINE_NAME[drawKey === 'model' ? 'physics' : 'model']}: torn off` : `${ENGINE_NAME[drawKey === 'model' ? 'physics' : 'model']}: stays on`);
    const tr = h('tr', { 'data-part': name, class: name === pinned ? 'pinned' : null },
      h('th', { scope: 'row', text: PART_LABEL[name] }),
      h('td', { class: 'bars' }, other ? [mk(main, ENGINE_COLOR[drawKey]), mk(other, ENGINE_COLOR[drawKey === 'model' ? 'physics' : 'model'])] : mk(main)),
      h('td', null, other ? [lvl(main), lvl(other)] : lvl(main)),
      h('td', { class: 'notes', text: notes.join(', ') }));
    // pointing at a row outlines that part in the drawing; a click keeps the outline (a tap on a phone)
    tr.addEventListener('pointerenter', () => { hovered = name; showOutline(); });
    tr.addEventListener('pointerleave', () => { hovered = null; showOutline(); });
    tr.addEventListener('click', () => {
      pinned = pinned === name ? null : name;
      tbody.querySelectorAll('tr').forEach((r) => r.classList.toggle('pinned', r.dataset.part === pinned));
      showOutline();
    });
    tbody.append(tr);
  }
  showOutline();
  const btn = $('#show-all');
  btn.textContent = state.showAll ? 'Show the 10 most damaged' : `Show all ${T.nParts} parts`;
  btn.setAttribute('aria-expanded', String(state.showAll));
  $('#parts').classList.toggle('compare', Boolean(other));
  $('#cmp-note').hidden = !other;
}

function showProblem(msg) {
  const p = $('#problem');
  p.hidden = false;
  p.textContent = `Check the inputs: ${msg}`;
  $('#results').classList.add('stale');
}
function hideProblem() {
  $('#problem').hidden = true;
  $('#results').classList.remove('stale');
}

// ---- start ----------------------------------------------------------------------------------------------------------

function buildStatic() {
  const fx = $('#effects');
  fx.textContent = '';
  for (const [key, label, why] of EFFECTS) {
    fx.append(h('li', { class: 'fx', 'data-fx': key, title: why }, h('span', { class: 'name', text: label }), h('span', { class: 'state', text: 'no' })));
  }
  const legend = $('#legend');
  legend.textContent = '';
  T.level_names.forEach((n, i) => legend.append(h('li', null, h('i', { class: 'sw', style: `background:${LEVEL_FILL[i]}` }), n)));
  $('#v-scale').textContent = '';
  for (let i = 0; i < 5; i++) $('#v-scale').append(h('i', { style: `--c:${LEVEL_FILL[Math.min(i + 0, 4)]}` }));
  document.querySelectorAll('#engine-switch input').forEach((i) => {
    if (!hasModel && i.value !== 'physics') { i.disabled = true; }
    i.addEventListener('change', () => { if (i.checked) { state.engine = i.value; if (i.value !== 'compare') state.drawEngine = i.value; schedule(); } });
  });
  document.querySelectorAll('#draw-switch input').forEach((i) => i.addEventListener('change', () => { if (i.checked) { state.drawEngine = i.value; schedule(); } }));
  $('#show-all').addEventListener('click', () => { state.showAll = !state.showAll; schedule(); });
  // the button text changes for a moment to say what happened; the button is read before the first await
  const flash = (btn, text) => {
    btn.dataset.label ??= btn.textContent; // remembered once, so a second click during the flash cannot freeze the new text
    btn.textContent = text;
    clearTimeout(btn.flashTimer);
    btn.flashTimer = setTimeout(() => { btn.textContent = btn.dataset.label; }, 1600);
  };
  const copyButton = (id, getText) => $(id).addEventListener('click', async (e) => {
    const btn = e.currentTarget;
    if (!lastRun?.req) return;
    flash(btn, (await copyText(getText())) ? 'Copied' : 'Copy failed');
  });
  copyButton('#copy-req', () => JSON.stringify(lastRun.req, null, 2));
  copyButton('#copy-rep', () => JSON.stringify(resultToReport(lastRun.sc, lastRun.out[lastRun.drawKey]), null, 2));
  copyButton('#copy-link', () => location.href);
  const m = hasModel ? T.modelInfo : null;
  if (m) $('#model-facts').textContent = m;
  scene = createScene($('#scene'));
  // on a phone the inputs sit below the result; this bar leads back up while they are being edited
  const mini = $('#mini'), narrow = window.matchMedia('(max-width: 999px)');
  let resultVisible = true;
  const showMini = () => { mini.hidden = resultVisible || !narrow.matches; };
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(([e]) => { resultVisible = e.isIntersecting; showMini(); }, { threshold: 0.02 }).observe($('#results'));
  }
  narrow.addEventListener?.('change', showMini);
  $('#mini-btn').addEventListener('click', () => {
    const calm = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    $('#results').scrollIntoView({ behavior: calm ? 'auto' : 'smooth', block: 'start' });
  });
}

async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function boot() {
  try {
    const [st, man] = await Promise.all([fetchJson('data/static.json'), fetchJson('data/model.json').catch(() => null)]);
    loadTables(st);
    if (man) {
      try {
        const buf = await (await fetch('data/model.bin')).arrayBuffer();
        loadModel(man, buf);
        hasModel = buf.byteLength === man.bytes;
        const mt = man.metrics;
        if (mt) T.modelInfo = `On 1,200 held-out scenarios the AI model is off by ${mt.damage_mae.toFixed(3)} on average per part (damage runs from 0 to 1) and picks the right damage level for ${(mt.level_acc * 100).toFixed(0)} percent of the parts, compared with the physics layer.`;
      } catch { hasModel = false; }
    }
  } catch (e) {
    document.body.classList.add('failed');
    $('#boot').textContent = `The data files could not be loaded (${e.message}). Open this page from the web address, not from a file.`;
    return;
  }
  document.body.classList.add('ready');
  $('#boot').hidden = true;
  if (!hasModel) { state.engine = 'physics'; state.drawEngine = 'physics'; $('#no-model').hidden = false; }
  resetCar();
  state.ob = newObstacle('concrete_wall');
  buildStatic();
  buildExamples();
  buildForm();
  if (!applyLink()) loadExample('side_pole_29kmh');
  run();
  // a link pasted into a tab that is already open only changes the hash, so that is handled here
  window.addEventListener('hashchange', () => { if (location.hash !== lastHash && applyLink()) run(); });
}

// Take the crash from the hash of the address, if it holds one; false when it does not.
function applyLink() {
  if (!readHash()) return false;
  if (!hasModel) state.engine = 'physics';
  state.example = null;
  if (state.engine !== 'compare') state.drawEngine = state.engine;
  document.querySelectorAll('#engine-switch input').forEach((i) => { i.checked = i.value === state.engine; });
  syncForm();
  syncExampleChips();
  return true;
}

boot();
