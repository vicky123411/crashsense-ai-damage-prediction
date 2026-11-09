// The drawing: the car (28 boxes) seen from above and from the left side, coloured by damage, with the obstacle,
// the contact patch, a target marker at the point of impact and the crush depth. Plain SVG, no libraries.
import { T } from './tables.js';
import { PART_LABEL, obstacleNoun, FACE_LABEL } from './describe.js';

const NS = 'http://www.w3.org/2000/svg';
const S = 100; // pixels per metre
const VB = { w: 620, h: 724 };
const CX = 310; // x = 0 m
const TOPV = { y0: 34, h: 360, cy: 214 }; // plan view: y runs +-1.8 m
const SIDEV = { y0: 432, h: 250, gz: 670 }; // side view: z from -0.12 m to 2.38 m
const X = (x) => CX + x * S;
const YT = (y) => TOPV.cy - y * S;
const ZS = (z) => SIDEV.gz - z * S;

export const LEVEL_FILL = ['#D5DEE6', '#F3D9A4', '#F0A04B', '#CF4125', '#7A1424'];
export const LEVEL_TEXT = ['#16222E', '#16222E', '#16222E', '#FFFFFF', '#FFFFFF'];
const INK = '#16222E', INK2 = '#4A5A69', YELLOW = '#FFC400', GRAPH = '#3B4650', GRID = '#DDE3E9';
const OB_TINT = { concrete: '#8E98A2', steel: '#5F7A92', wood: '#A47E4E', foam: '#7FB3BF' };
// outline of the obstacle by its own damage level (darker than the fills used for the car, to stay visible)
const OB_STROKE = [INK, '#B07A12', '#C4631A', '#CF4125', '#7A1424'];

function el(name, attrs = {}, parent = null, text = null) {
  const e = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text !== null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}

const fmt = (x) => (Math.round(x * 10) / 10).toString();

export function createScene(svg) {
  svg.setAttribute('viewBox', `0 0 ${VB.w} ${VB.h}`);
  svg.setAttribute('role', 'img');
  svg.textContent = '';
  const title = el('title', { id: 'scene-title' }, svg, 'Damage drawing');
  const desc = el('desc', { id: 'scene-desc' }, svg, '');
  svg.setAttribute('aria-labelledby', 'scene-title scene-desc');

  const defs = el('defs', {}, svg);
  for (const [id, y0, h] of [['clipTop', TOPV.y0, TOPV.h], ['clipSide', SIDEV.y0, SIDEV.h]]) {
    el('rect', { x: 0, y: y0, width: VB.w, height: h }, el('clipPath', { id }, defs));
  }
  const hatch = el('pattern', { id: 'hatch', width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
  el('line', { x1: 0, y1: 0, x2: 0, y2: 7, stroke: INK, 'stroke-width': 1.6, opacity: 0.55 }, hatch);
  const ground = el('pattern', { id: 'ground', width: 8, height: 8, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(-45)' }, defs);
  el('line', { x1: 0, y1: 0, x2: 0, y2: 8, stroke: INK2, 'stroke-width': 1, opacity: 0.45 }, ground);

  // ---- sheets with grid ----
  const panels = [
    { id: 'top', y0: TOPV.y0, h: TOPV.h, label: 'Top view', clip: 'clipTop' },
    { id: 'side', y0: SIDEV.y0, h: SIDEV.h, label: "Side view, from the car's left", clip: 'clipSide' },
  ];
  const sheets = el('g', { class: 'sheets' }, svg);
  for (const p of panels) {
    el('rect', { x: 0.5, y: p.y0, width: VB.w - 1, height: p.h, fill: '#FAFBFC', stroke: '#CBD3DB' }, sheets);
    const g = el('g', { 'clip-path': `url(#${p.clip})` }, sheets);
    for (let m = -3; m <= 3; m += 0.5) el('line', { x1: X(m), y1: p.y0, x2: X(m), y2: p.y0 + p.h, stroke: GRID, 'stroke-width': m % 1 === 0 ? 1.2 : 0.6 }, g);
    el('text', { x: 6, y: p.y0 - 8, class: 'pt' }, sheets, p.label);
  }
  const gTopGrid = el('g', { 'clip-path': 'url(#clipTop)' }, sheets);
  for (let m = -1.5; m <= 1.5; m += 0.5) el('line', { x1: 0, y1: YT(m), x2: VB.w, y2: YT(m), stroke: GRID, 'stroke-width': m % 1 === 0 ? 1.2 : 0.6 }, gTopGrid);
  const gSideGrid = el('g', { 'clip-path': 'url(#clipSide)' }, sheets);
  for (let m = 0.5; m <= 2.3; m += 0.5) el('line', { x1: 0, y1: ZS(m), x2: VB.w, y2: ZS(m), stroke: GRID, 'stroke-width': m % 1 === 0 ? 1.2 : 0.6 }, gSideGrid);
  el('rect', { x: 0.5, y: ZS(0), width: VB.w - 1, height: SIDEV.y0 + SIDEV.h - ZS(0), fill: 'url(#ground)' }, sheets);
  el('line', { x1: 0.5, y1: ZS(0), x2: VB.w - 0.5, y2: ZS(0), stroke: INK2, 'stroke-width': 1.6 }, sheets);

  // ---- car outline (where the car was) ----
  const L = T.constants.CAR_LENGTH, Wd = T.constants.CAR_WIDTH, H = T.constants.CAR_HEIGHT;
  const outline = { fill: 'none', stroke: '#8795A3', 'stroke-width': 1, 'stroke-dasharray': '6 4' };
  el('rect', { x: X(-L / 2), y: YT(Wd / 2), width: L * S, height: Wd * S, ...outline }, svg);
  el('rect', { x: X(-L / 2), y: ZS(H), width: L * S, height: H * S, ...outline }, svg);

  // ---- layers ----
  const obstacleBehind = el('g', { 'clip-path': 'url(#clipSide)' }, svg); // obstacle hidden behind the car in the side view
  const gTop = el('g', { 'clip-path': 'url(#clipTop)' }, svg);
  const gSide = el('g', { 'clip-path': 'url(#clipSide)' }, svg);
  const partNodes = { top: {}, side: {} };
  const order = T.parts.map((p, i) => i);
  const topOrder = [...order].sort((a, b) => (T.parts[a].center[2] + T.parts[a].size[2] / 2) - (T.parts[b].center[2] + T.parts[b].size[2] / 2));
  const sideOrder = [...order].sort((a, b) => T.parts[a].center[1] - T.parts[b].center[1]); // right side first, left side on top
  const makePart = (i, view, parent) => {
    const p = T.parts[i];
    const g = el('g', { class: 'part', 'data-part': p.name }, parent);
    const [cx, cy, cz] = p.center, [lx, ly, lz] = p.size;
    const r = view === 'top'
      ? { x: X(cx - lx / 2), y: YT(cy + ly / 2), width: lx * S, height: ly * S }
      : { x: X(cx - lx / 2), y: ZS(cz + lz / 2), width: lx * S, height: lz * S };
    const rx = p.kind === 'wheel' ? 5 : 1.5;
    const rect = el('rect', { ...r, rx, 'stroke-width': 1 }, g);
    // a part that tears off gets a white and black dashed outline, which shows on every fill colour
    const ring = el('rect', { ...r, rx, fill: 'none', stroke: INK, 'stroke-width': 2, 'stroke-dasharray': '4 4', visibility: 'hidden' }, g);
    const tip = el('title', {}, g, '');
    const mid = [r.x + r.width / 2, r.y + r.height / 2];
    partNodes[view][p.name] = { g, rect, ring, tip, mid };
  };
  topOrder.forEach((i) => makePart(i, 'top', gTop));
  sideOrder.forEach((i) => makePart(i, 'side', gSide));

  const obstacleTop = el('g', { 'clip-path': 'url(#clipTop)' }, svg);
  const obstacleSide = el('g', { 'clip-path': 'url(#clipSide)' }, svg);
  const annTop = el('g', { 'clip-path': 'url(#clipTop)' }, svg);
  const annSide = el('g', { 'clip-path': 'url(#clipSide)' }, svg);
  const fixedLabels = el('g', {}, svg);
  el('text', { x: X(L / 2 - 0.05), y: YT(-Wd / 2) + 18, class: 'dl halo', 'text-anchor': 'end' }, fixedLabels, 'Front');
  el('text', { x: X(-L / 2 + 0.05), y: YT(-Wd / 2) + 18, class: 'dl halo', 'text-anchor': 'start' }, fixedLabels, 'Rear');
  const block = el('g', { class: 'block' }, svg);
  // outline of the part that is picked in the table; drawn on top of everything else
  const hlTop = el('g', { 'clip-path': 'url(#clipTop)', 'pointer-events': 'none' }, svg);
  const hlSide = el('g', { 'clip-path': 'url(#clipSide)', 'pointer-events': 'none' }, svg);

  function placeTarget(parent, x, y) {
    const g = el('g', {}, parent);
    const r = 11;
    el('circle', { cx: x, cy: y, r: r + 2, fill: '#fff' }, g);
    el('path', { d: `M${x} ${y} L${x} ${y - r} A${r} ${r} 0 0 1 ${x + r} ${y} Z`, fill: INK }, g);
    el('path', { d: `M${x} ${y} L${x + r} ${y} A${r} ${r} 0 0 1 ${x} ${y + r} Z`, fill: YELLOW }, g);
    el('path', { d: `M${x} ${y} L${x} ${y + r} A${r} ${r} 0 0 1 ${x - r} ${y} Z`, fill: INK }, g);
    el('path', { d: `M${x} ${y} L${x - r} ${y} A${r} ${r} 0 0 1 ${x} ${y - r} Z`, fill: YELLOW }, g);
    el('circle', { cx: x, cy: y, r, fill: 'none', stroke: INK, 'stroke-width': 1.2 }, g);
    for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
      el('line', { x1: x + dx * (r + 2), y1: y + dy * (r + 2), x2: x + dx * (r + 12), y2: y + dy * (r + 12), stroke: INK, 'stroke-width': 1.4 }, g);
    }
  }

  function arrowLine(parent, x1, y1, x2, y2, color, width = 1.6) {
    el('line', { x1, y1, x2, y2, stroke: color, 'stroke-width': width }, parent);
    const ang = Math.atan2(y2 - y1, x2 - x1), a = 7;
    for (const [px, py, dir] of [[x1, y1, 1], [x2, y2, -1]]) {
      const a1 = ang + (dir > 0 ? 0 : Math.PI);
      el('path', { d: `M${px} ${py} L${px + a * Math.cos(a1 + 0.4)} ${py + a * Math.sin(a1 + 0.4)} L${px + a * Math.cos(a1 - 0.4)} ${py + a * Math.sin(a1 - 0.4)} Z`, fill: color }, parent);
    }
  }

  // an obstacle shape as an SVG node; (u0,u1) range along the push axis, (a0,a1) across, in px, view-specific
  function obstacleShape(parent, ob, rect, opts) {
    const tint = OB_TINT[ob.material] || GRAPH;
    const common = { fill: tint, 'fill-opacity': opts.light ? 0.25 : 0.55, stroke: opts.stroke, 'stroke-width': opts.sw };
    let node;
    if (opts.wedge) {
      node = el('polygon', { points: opts.wedge, ...common }, parent);
    } else {
      node = el('rect', { x: rect.x, y: rect.y, width: Math.max(rect.w, 1.5), height: Math.max(rect.h, 1.5),
        rx: ob.shape === 'round' ? Math.min(rect.w, rect.h) / 2 : 0, ...common }, parent);
    }
    // hatch on top so the obstacle never looks like a car part
    const hat = node.cloneNode(false);
    hat.setAttribute('fill', 'url(#hatch)');
    hat.setAttribute('fill-opacity', 1);
    hat.setAttribute('stroke', 'none');
    parent.appendChild(hat);
    return node;
  }

  function update(sc, res, info = {}) {
    const face = sc.face, geo = res.geo, ob = sc.obstacle;
    const s = geo.sign, ax = geo.ax;
    const plane = T.face_plane[face];
    const half = T.face_half_span[face];
    const crush = Math.min(Math.max(res.crush_depth, 0), T.extent[face]);
    // ---- parts ----
    T.parts.forEach((p, i) => {
      const lvl = res.level[i], torn = Boolean(res.detached[i]);
      const label = `${PART_LABEL[p.name]}: ${T.level_names[lvl]}, damage ${res.damage[i].toFixed(2)}${torn ? ', torn off' : ''}`;
      for (const view of ['top', 'side']) {
        const n = partNodes[view][p.name];
        n.rect.setAttribute('fill', LEVEL_FILL[lvl]);
        n.rect.setAttribute('stroke', torn ? '#FFFFFF' : INK);
        n.rect.setAttribute('stroke-width', torn ? 2 : 1);
        n.ring.setAttribute('visibility', torn ? 'visible' : 'hidden');
        n.tip.textContent = label;
        if (torn) {
          // thrown off the car: pushed away from the middle of the car and tilted
          const c = view === 'top' ? [CX, TOPV.cy] : [CX, ZS(0.72)];
          let dx = n.mid[0] - c[0], dy = n.mid[1] - c[1];
          const len = Math.hypot(dx, dy) || 1;
          dx = (dx / len) * 34; dy = (dy / len) * 34;
          const tilt = ((i * 37) % 21) - 10;
          n.g.setAttribute('transform', `translate(${dx.toFixed(1)} ${dy.toFixed(1)}) rotate(${tilt} ${n.mid[0].toFixed(1)} ${n.mid[1].toFixed(1)})`);
        } else n.g.removeAttribute('transform');
      }
    });
    for (const g of [obstacleTop, obstacleSide, obstacleBehind, annTop, annSide, block, hlTop, hlSide]) g.textContent = '';

    // ---- obstacle ----
    const depthEff = res.obstacle_depth !== null && res.obstacle_depth !== undefined
      ? Math.max(ob.depth - Math.min(res.obstacle_depth, 0.9 * ob.depth), 0.01) : ob.depth;
    const cs = s * (plane - crush); // contact surface, along the push axis
    const p0 = Math.min(cs, cs + s * depthEff), p1 = Math.max(cs, cs + s * depthEff);
    const c = geo.centre, a0 = c - ob.width / 2, a1 = c + ob.width / 2;
    const z0 = ob.z_base, z1 = ob.z_base + ob.height;
    const stroke = OB_STROKE[res.obstacle_level] || INK;
    const sw = res.obstacle_level > 0 ? 2.6 : 1.8;
    // plan view rectangle in world metres: x range, y range
    const topBox = ax === 0 ? { x0: p0, x1: p1, y0: a0, y1: a1 } : { x0: a0, x1: a1, y0: p0, y1: p1 };
    const rectTop = { x: X(topBox.x0), y: YT(topBox.y1), w: (topBox.x1 - topBox.x0) * S, h: (topBox.y1 - topBox.y0) * S };
    let wedgeTop = null;
    if (ob.shape === 'sharp') { // a wedge whose point touches the car
      const baseS = cs + s * depthEff;
      wedgeTop = ax === 0
        ? `${X(cs)},${YT(c)} ${X(baseS)},${YT(a1)} ${X(baseS)},${YT(a0)}`
        : `${X(c)},${YT(cs)} ${X(a1)},${YT(baseS)} ${X(a0)},${YT(baseS)}`;
    }
    const sideBox = ax === 0 ? { x0: p0, x1: p1 } : { x0: a0, x1: a1 };
    const rectSide = { x: X(sideBox.x0), y: ZS(z1), w: (sideBox.x1 - sideBox.x0) * S, h: ob.height * S };
    const behind = face === 'right';
    obstacleShape(obstacleTop, ob, rectTop, { stroke, sw, wedge: wedgeTop });
    obstacleShape(behind ? obstacleBehind : obstacleSide, ob, rectSide, { stroke, sw, light: face === 'left' });
    // name next to the obstacle (plan view)
    const nameAt = ax === 0
      ? { x: s > 0 ? X(p1) + 8 : X(p0) - 8, y: YT(c) - Math.min(ob.width, 1.4) * S / 2 - 8, anchor: s > 0 ? 'start' : 'end' }
      : { x: X(c), y: s > 0 ? YT(p1) - 10 : YT(p0) + 20, anchor: 'middle' };
    const nm = el('text', { x: nameAt.x, y: Math.max(nameAt.y, TOPV.y0 + 16), class: 'dl halo', 'text-anchor': nameAt.anchor }, annTop, obstacleNoun(ob));

    // ---- contact patch, cabin line, crush ----
    const lo = geo.lo, hi = geo.hi;
    const patch = { stroke: YELLOW, 'stroke-width': 6, 'stroke-linecap': 'butt' };
    if (ax === 0) el('line', { x1: X(s * plane), y1: YT(lo), x2: X(s * plane), y2: YT(hi), ...patch }, annTop);
    else el('line', { x1: X(lo), y1: YT(s * plane), x2: X(hi), y2: YT(s * plane), ...patch }, annTop);
    // cabin line
    const cabin = s * (plane - T.cabin_depth[face]);
    const dot = { stroke: INK2, 'stroke-width': 1.2, 'stroke-dasharray': '2 4', 'stroke-linecap': 'round' };
    if (ax === 0) {
      el('line', { x1: X(cabin), y1: YT(-half), x2: X(cabin), y2: YT(half), ...dot }, annTop);
      el('text', { x: X(cabin) + (s > 0 ? -5 : 5), y: YT(half) + 14, class: 'dl small halo', 'text-anchor': s > 0 ? 'end' : 'start' }, annTop, 'Passenger cell');
    } else {
      el('line', { x1: X(-T.face_half_span[face]), y1: YT(cabin), x2: X(T.face_half_span[face]), y2: YT(cabin), ...dot }, annTop);
      el('text', { x: X(-T.face_half_span[face]) + 6, y: YT(cabin) + (s > 0 ? 14 : -6), class: 'dl small halo' }, annTop, 'Passenger cell');
    }
    // crush dimension line beside the patch
    if (crush > 0.01) {
      const ap = hi + 0.14 <= half ? hi + 0.14 : (lo - 0.14 >= -half ? lo - 0.14 : half - 0.12);
      const pA = s * plane, pB = s * (plane - crush), pCab = s * (plane - Math.min(crush, T.cabin_depth[face]));
      const pt = (p, a) => (ax === 0 ? [X(p), YT(a)] : [X(a), YT(p)]);
      const [x1, y1] = pt(pA, ap), [x2, y2] = pt(pB, ap);
      arrowLine(annTop, x1, y1, x2, y2, INK, 1.4);
      if (crush > T.cabin_depth[face] + 0.005) {
        const [xc, yc] = pt(pCab, ap);
        el('line', { x1: xc, y1: yc, x2, y2, stroke: '#CF4125', 'stroke-width': 5, 'stroke-linecap': 'butt', opacity: 0.9 }, annTop);
        arrowLine(annTop, xc, yc, x2, y2, '#7A1424', 1.4);
      }
    }
    // target marker where the hit lands
    placeTarget(annTop, X(geo.point[0]), YT(geo.point[1]));
    placeTarget(annSide, X(geo.point[0]), ZS(geo.point[2]));
    // patch in the side view
    const sp = { fill: YELLOW, 'fill-opacity': 0.35, stroke: YELLOW, 'stroke-width': 2.5 };
    if (ax === 0) el('line', { x1: X(s * plane), y1: ZS(geo.z0), x2: X(s * plane), y2: ZS(geo.z1), ...patch }, annSide);
    else el('rect', { x: X(lo), y: ZS(geo.z1), width: (hi - lo) * S, height: (geo.z1 - geo.z0) * S, ...sp, ...(face === 'right' ? { 'stroke-dasharray': '5 3', 'fill-opacity': 0.15 } : {}) }, annSide);

    // ---- title block ----
    el('line', { x1: 0.5, y1: 694, x2: VB.w - 0.5, y2: 694, stroke: '#CBD3DB' }, block);
    el('text', { x: 6, y: 714, class: 'tb' }, block, `${FACE_LABEL[face]} hit on a ${obstacleNoun(ob)}, ${Math.round(sc.speed * 3.6)} km/h`);
    const eng = info.label || '';
    if (eng) { // a dot in the engine's colour, placed just left of the measured text
      const t = el('text', { x: VB.w - 6, y: 714, class: 'tb', 'text-anchor': 'end' }, block, eng);
      const w = (t.getComputedTextLength && t.getComputedTextLength()) || eng.length * 8;
      el('circle', { cx: VB.w - 6 - w - 11, cy: 709, r: 5, fill: info.color || INK }, block);
    }
    // scale bar
    el('line', { x1: X(-3), y1: 683, x2: X(-2), y2: 683, stroke: INK, 'stroke-width': 2 }, block);
    el('text', { x: X(-2) + 5, y: 687, class: 'dl small' }, block, '1 m');
    desc.textContent = info.alt || '';
    title.textContent = 'Damage drawing';
  }

  // Outline one part in both views (name = null clears it). It is an overlay, so the stacking order of the parts is untouched.
  function highlight(name) {
    hlTop.textContent = '';
    hlSide.textContent = '';
    if (!name) return;
    for (const [view, layer] of [['top', hlTop], ['side', hlSide]]) {
      const n = partNodes[view][name];
      if (!n) continue;
      const geom = ['x', 'y', 'width', 'height', 'rx'].map((k) => [k, n.rect.getAttribute(k)]);
      const moved = n.g.getAttribute('transform');
      for (const [stroke, width] of [[INK, 7], [YELLOW, 3.5]]) {
        const o = el('rect', { ...Object.fromEntries(geom), fill: 'none', stroke, 'stroke-width': width }, layer);
        if (moved) o.setAttribute('transform', moved);
      }
    }
  }

  return { update, highlight };
}
