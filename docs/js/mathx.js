// Small numeric helpers that make the JavaScript code give the same numbers as the Python (numpy / torch) code.

// Sum of a list the way numpy adds a float64 row: blocks of eight running sums, then combined (pairwise).
export function sumNp(a) {
  const n = a.length;
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[i];
    return res;
  }
  if (n <= 128) {
    const r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) for (let j = 0; j < 8; j++) r[j] += a[i + j];
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[i];
    return res;
  }
  let n2 = n >> 1;
  n2 -= n2 % 8;
  return sumNp(a.slice(0, n2)) + sumNp(a.slice(n2));
}

// Error function, accurate to about 1e-15 (Taylor series for small |x|, continued fraction for the tail).
export function erf(x) {
  const ax = Math.abs(x);
  if (ax < 2.5) {
    let term = ax, sum = ax;
    const x2 = ax * ax;
    for (let n = 1; n < 80; n++) {
      term *= -x2 / n;
      const add = term / (2 * n + 1);
      sum += add;
      if (Math.abs(add) < 1e-17 * Math.abs(sum)) break;
    }
    return Math.sign(x) * (2 / Math.sqrt(Math.PI)) * sum;
  }
  if (ax > 6) return Math.sign(x);
  // erfc(x) = exp(-x^2) / sqrt(pi) * 1 / (x + (1/2) / (x + 1 / (x + (3/2) / (x + 2 / (x + ...)))))
  let f = ax;
  for (let k = 60; k >= 1; k--) f = ax + (k / 2) / f;
  const erfc = Math.exp(-ax * ax) / (Math.sqrt(Math.PI) * f);
  return Math.sign(x) * (1 - erfc);
}

// The exact GELU that torch.nn.GELU() uses.
export const gelu = (x) => 0.5 * x * (1 + erf(x * Math.SQRT1_2));
export const sigmoid = (x) => 1 / (1 + Math.exp(-x));

// Python's math.radians
export const radians = (deg) => deg * (Math.PI / 180);

// Python's round(x, n): round half to even on the exact value.
export function pyRound(x, n) {
  if (!Number.isFinite(x)) return x;
  const s = x.toFixed(n);
  return Number(s);
}
