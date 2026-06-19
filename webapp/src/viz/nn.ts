// A tiny multilayer perceptron with backprop, written from scratch (mirrors playbook P11).
// Used by the Neural Network Playground to train in-browser on 2-D toy datasets.

export type Act = "tanh" | "relu" | "sigmoid";

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));
const actFns: Record<Act, (z: number) => number> = {
  tanh: Math.tanh,
  relu: (z) => Math.max(0, z),
  sigmoid,
};
const actDeriv: Record<Act, (z: number, a: number) => number> = {
  tanh: (_z, a) => 1 - a * a,
  relu: (z) => (z > 0 ? 1 : 0),
  sigmoid: (_z, a) => a * (1 - a),
};

type Layer = {
  W: number[][]; // [out][in]
  b: number[];   // [out]
  // caches for backprop
  z: number[];
  a: number[];
  input: number[];
};

const randn = () => {
  // Box–Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

export class MLP {
  layers: Layer[] = [];
  act: Act;
  constructor(sizes: number[], act: Act) {
    this.act = act;
    for (let i = 0; i < sizes.length - 1; i++) {
      const inN = sizes[i], outN = sizes[i + 1];
      const scale = Math.sqrt(2 / inN); // He-ish init
      this.layers.push({
        W: Array.from({ length: outN }, () => Array.from({ length: inN }, () => randn() * scale)),
        b: Array.from({ length: outN }, () => 0),
        z: [], a: [], input: [],
      });
    }
  }

  // forward; final layer uses sigmoid (binary output), hidden layers use this.act
  forward(x: number[]): number {
    let a = x;
    this.layers.forEach((L, li) => {
      L.input = a;
      const isLast = li === this.layers.length - 1;
      const fn = isLast ? sigmoid : actFns[this.act];
      L.z = L.W.map((row, j) => row.reduce((s, w, k) => s + w * a[k], 0) + L.b[j]);
      L.a = L.z.map(fn);
      a = L.a;
    });
    return a[0];
  }

  // one SGD step on a batch; returns mean binary cross-entropy loss
  trainStep(X: number[][], Y: number[], lr: number): number {
    // accumulate gradients
    const gW = this.layers.map((L) => L.W.map((row) => row.map(() => 0)));
    const gB = this.layers.map((L) => L.b.map(() => 0));
    let loss = 0;
    const nLast = this.layers.length - 1;

    for (let n = 0; n < X.length; n++) {
      const out = this.forward(X[n]);
      const p = Math.min(Math.max(out, 1e-7), 1 - 1e-7);
      const y = Y[n];
      loss += -(y * Math.log(p) + (1 - y) * Math.log(1 - p));

      // backprop: delta for each layer
      const deltas: number[][] = this.layers.map((L) => L.z.map(() => 0));
      // output layer: sigmoid + BCE -> delta = (p - y)
      deltas[nLast][0] = p - y;
      for (let li = nLast - 1; li >= 0; li--) {
        const Lnext = this.layers[li + 1];
        const L = this.layers[li];
        for (let j = 0; j < L.z.length; j++) {
          let upstream = 0;
          for (let k = 0; k < Lnext.z.length; k++) upstream += Lnext.W[k][j] * deltas[li + 1][k];
          deltas[li][j] = upstream * actDeriv[this.act](L.z[j], L.a[j]);
        }
      }
      // accumulate weight/bias grads
      for (let li = 0; li < this.layers.length; li++) {
        const L = this.layers[li];
        for (let j = 0; j < L.W.length; j++) {
          gB[li][j] += deltas[li][j];
          for (let k = 0; k < L.W[j].length; k++) gW[li][j][k] += deltas[li][j] * L.input[k];
        }
      }
    }

    // apply averaged gradients
    const m = X.length;
    for (let li = 0; li < this.layers.length; li++) {
      const L = this.layers[li];
      for (let j = 0; j < L.W.length; j++) {
        L.b[j] -= (lr * gB[li][j]) / m;
        for (let k = 0; k < L.W[j].length; k++) L.W[j][k] -= (lr * gW[li][j][k]) / m;
      }
    }
    return loss / m;
  }
}

// ---- 2-D toy datasets (label 0 / 1), coordinates in [-1, 1] -----------------
export type Dataset = "circle" | "xor" | "gauss" | "spiral";
export type Point = { x: number; y: number; label: number };

export function makeData(kind: Dataset, n = 200, noise = 0.1): Point[] {
  const pts: Point[] = [];
  const jitter = () => (Math.random() - 0.5) * noise;
  if (kind === "circle") {
    for (let i = 0; i < n; i++) {
      const inner = i < n / 2;
      const r = inner ? Math.random() * 0.4 : 0.55 + Math.random() * 0.45;
      const t = Math.random() * 2 * Math.PI;
      pts.push({ x: r * Math.cos(t) + jitter(), y: r * Math.sin(t) + jitter(), label: inner ? 1 : 0 });
    }
  } else if (kind === "xor") {
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
      pts.push({ x: x + jitter(), y: y + jitter(), label: x * y > 0 ? 1 : 0 });
    }
  } else if (kind === "gauss") {
    for (let i = 0; i < n; i++) {
      const a = i < n / 2;
      const cx = a ? 0.5 : -0.5, cy = a ? 0.5 : -0.5;
      pts.push({ x: cx + randn() * 0.18 + jitter(), y: cy + randn() * 0.18 + jitter(), label: a ? 1 : 0 });
    }
  } else {
    // spiral
    for (let i = 0; i < n; i++) {
      const a = i < n / 2;
      const idx = i % (n / 2);
      const r = (idx / (n / 2)) * 0.9;
      const t = (idx / (n / 2)) * 3.2 + (a ? 0 : Math.PI);
      pts.push({ x: r * Math.cos(t) + jitter(), y: r * Math.sin(t) + jitter(), label: a ? 1 : 0 });
    }
  }
  return pts;
}
