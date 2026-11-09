// Checks that the JavaScript copy of the engine (docs/js) gives the same numbers as the Python code.
//   node tests/web/check_parity.mjs [cases.json]
// Cases come from scripts/make_web_parity.py. Exit code 1 if anything is off.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { loadTables, T } from '../../docs/js/tables.js';
import { erf } from '../../docs/js/mathx.js';
import { scenarioFromDict, simulate, contactGeometry, resultToReport } from '../../docs/js/physics.js';
import { scenarioInputs } from '../../docs/js/features.js';
import { loadModel, predictModel, predictArrays } from '../../docs/js/gnn.js';

const here = dirname(fileURLToPath(import.meta.url));
const data = join(here, '..', '..', 'docs', 'data');
const casesPath = process.argv[2] || join(here, 'parity_cases.json');

loadTables(JSON.parse(readFileSync(join(data, 'static.json'), 'utf8')));
const bin = readFileSync(join(data, 'model.bin'));
loadModel(JSON.parse(readFileSync(join(data, 'model.json'), 'utf8')),
  bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength));
const { cases, erf: erfTable } = JSON.parse(readFileSync(casesPath, 'utf8'));

let failures = 0;
const worst = {}; // largest deviation per quantity
const fail = (msg) => { failures++; if (failures <= 25) console.log('FAIL', msg); };

// relative + absolute deviation of two numbers
const dev = (a, b) => Math.abs(a - b) / (1e-12 + Math.abs(b));
function num(tag, got, want, tol, i) {
  if (got === null || got === undefined || want === null || want === undefined) {
    if (!(got == null && want == null)) fail(`case ${i} ${tag}: ${got} vs ${want}`);
    return;
  }
  const d = Math.min(Math.abs(got - want), dev(got, want));
  worst[tag] = Math.max(worst[tag] || 0, d);
  if (!(d <= tol)) fail(`case ${i} ${tag}: got ${got}, want ${want}`);
}
function arr(tag, got, want, tol, i) {
  if (got.length !== want.length) return fail(`case ${i} ${tag}: length ${got.length} vs ${want.length}`);
  for (let k = 0; k < want.length; k++) num(`${tag}`, Number(got[k]), Number(want[k]), tol, `${i}[${k}]`);
}
function same(tag, got, want, i) {
  if (JSON.stringify(got) !== JSON.stringify(want)) fail(`case ${i} ${tag}: ${JSON.stringify(got)} vs ${JSON.stringify(want)}`);
}

// error function used by the network's GELU
for (let k = 0; k < erfTable.x.length; k++) num('erf', erf(erfTable.x[k]), erfTable.y[k], 1e-14, `x=${erfTable.x[k]}`);

const PHYS = 1e-9;  // physics layer: both sides use IEEE doubles and the same operations
const MODEL = 2e-4; // network: float32 in Python, float64 here
let nModelDiffs = 0;
cases.forEach((c, i) => {
  const e = c.expected;
  const sc = scenarioFromDict(c.request);
  const res = simulate(sc);
  const p = e.physics;
  arr('damage', res.damage, p.damage, PHYS, i);
  same('level', Array.from(res.level), p.level, i);
  same('detached', Array.from(res.detached), p.detached, i);
  same('direct', Array.from(res.direct), p.direct, i);
  for (const k of ['crush_depth', 'intrusion', 'peak_force_kn', 'e_diss_kj', 'e_car_kj', 'e_obstacle_kj', 'unabsorbed_kj',
    'obstacle_damage', 'obstacle_depth', 'delta_v', 'obstacle_speed_after']) num(k, res[k], p[k], PHYS, i);
  same('obstacle_level', res.obstacle_level, p.obstacle_level, i);
  for (const k of ['breakthrough', 'through', 'airbag_front', 'airbag_side']) same(k, res[k], p[k], i);
  same('effects', res.effects, p.effects, i);
  arr('capacity_kj', res.capacity_kj, p.capacity_kj, PHYS, i);

  const geo = contactGeometry(sc);
  for (const k of ['overlap', 'depth_start', 'stroke', 'dist', 'point']) arr(`geo.${k}`, geo[k], e.geo[k], PHYS, i);
  for (const k of ['width', 'centre', 'lo', 'hi', 'z0', 'z1', 'arm']) num(`geo.${k}`, geo[k], e.geo[k], PHYS, i);

  // report in the Unity JSON layout (values are rounded to 4 digits, so one rounding step of slack)
  const rep = resultToReport(sc, res), want = e.physics_report;
  same('report.keys', Object.keys(rep.summary), Object.keys(want.summary), i);
  for (const k of Object.keys(want.summary)) {
    if (typeof want.summary[k] === 'number') num('report.' + k, rep.summary[k], want.summary[k], 1.1e-4, i);
    else same('report.' + k, rep.summary[k], want.summary[k], i);
  }
  rep.parts.forEach((part, k) => {
    num('report.part.damage', part.damage, want.parts[k].damage, 1.1e-4, `${i}[${k}]`);
    same('report.part.level', [part.level, part.detached, part.direct], [want.parts[k].level, want.parts[k].detached, want.parts[k].direct], `${i}[${k}]`);
  });
  same('report.effects', rep.effects, want.effects, i);

  // the network
  const inp = scenarioInputs(sc, geo);
  if (e.features) {
    arr('features.node', inp.node, e.features.node.flat(), 2e-6, i);
    arr('features.glob', inp.glob, e.features.glob, 2e-6, i);
  }
  const out = predictArrays(inp.node, inp.glob);
  arr('model.damage', out.damage, e.model.damage, MODEL, i);
  arr('model.detach_prob', out.detachProb, e.model.detach_prob, MODEL, i);
  arr('model.reg', out.reg, e.model.reg, MODEL, i);
  arr('model.flags_prob', out.flags, e.model.flags_prob, MODEL, i);
  const mres = predictModel(sc), mrep = resultToReport(sc, mres), mwant = e.model_report;
  same('model_report.summary.keys', Object.keys(mrep.summary), Object.keys(mwant.summary), i);
  for (const k of Object.keys(mwant.summary)) {
    if (typeof mwant.summary[k] === 'number') num('model_report.' + k, mrep.summary[k], mwant.summary[k], 1e-3, i);
    else if (mrep.summary[k] !== mwant.summary[k]) nModelDiffs++; // a yes/no flag that sits exactly on 0.5
  }
  same('model_report.obstacle.keys', Object.keys(mrep.obstacle), Object.keys(mwant.obstacle), i);
  mrep.parts.forEach((part, k) => {
    if (Math.abs(part.damage - mwant.parts[k].damage) > 1e-3) fail(`case ${i} model part ${k} damage ${part.damage} vs ${mwant.parts[k].damage}`);
    if (part.detached !== mwant.parts[k].detached) nModelDiffs++;
  });
});

console.log(`${cases.length} cases checked, ${failures} failures.`);
console.log('largest deviation per quantity (relative or absolute, whichever is smaller):');
for (const [k, v] of Object.entries(worst).sort((a, b) => b[1] - a[1]).slice(0, 12)) console.log(`  ${k.padEnd(22)} ${v.toExponential(2)}`);
if (nModelDiffs) console.log(`yes/no model outputs that differ because they sit on the 0.5 line: ${nModelDiffs}`);
process.exit(failures ? 1 : 0);
