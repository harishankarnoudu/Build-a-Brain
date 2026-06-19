import { useEffect, useRef, useState } from "react";
import { VizCard, Slider, Readout } from "../components/ui";

// Fit a polynomial of adjustable degree to noisy points; watch train vs test error diverge.
const W = 600, H = 340, PAD = 30;
const trueF = (x: number) => 0.5 * Math.sin(1.5 * x) + 0.12 * x;

// deterministic noisy training sample
const TRAIN = (() => {
  let s = 7; const rnd = () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
  const pts: [number, number][] = [];
  for (let i = 0; i < 11; i++) { const x = -2.8 + (i / 10) * 5.6; pts.push([x, trueF(x) + (rnd() - 0.5) * 0.9]); }
  return pts;
})();

// least-squares polynomial fit (x scaled to [-1,1] for stability); returns coeffs ascending
function polyfit(pts: [number, number][], degree: number): number[] {
  const sx = (x: number) => x / 3;
  const n = degree + 1;
  const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const c: number[] = new Array(n).fill(0);
  for (const [xr, y] of pts) {
    const x = sx(xr); const pw: number[] = [1];
    for (let k = 1; k < 2 * n; k++) pw.push(pw[k - 1] * x);
    for (let i = 0; i < n; i++) { c[i] += pw[i] * y; for (let j = 0; j < n; j++) A[i][j] += pw[i + j]; }
  }
  // Gaussian elimination with partial pivoting
  for (let i = 0; i < n; i++) {
    let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
    [A[i], A[p]] = [A[p], A[i]]; [c[i], c[p]] = [c[p], c[i]];
    if (Math.abs(A[i][i]) < 1e-12) continue;
    for (let r = 0; r < n; r++) if (r !== i) { const f = A[r][i] / A[i][i]; for (let k = i; k < n; k++) A[r][k] -= f * A[i][k]; c[r] -= f * c[i]; }
  }
  return A.map((row, i) => c[i] / (row[i] || 1));
}
const evalPoly = (co: number[], x: number) => { const s = x / 3; let r = 0, p = 1; for (const k of co) { r += k * p; p *= s; } return r; };

export default function OverfittingExplorer() {
  const [degree, setDegree] = useState(3);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const co = polyfit(TRAIN, degree);
  const trainMSE = TRAIN.reduce((s, [x, y]) => s + (evalPoly(co, x) - y) ** 2, 0) / TRAIN.length;
  let testMSE = 0; for (let i = 0; i <= 60; i++) { const x = -2.8 + (i / 60) * 5.6; testMSE += (evalPoly(co, x) - trueF(x)) ** 2; } testMSE /= 61;

  const px = (x: number) => PAD + ((x + 3) / 6) * (W - 2 * PAD);
  const py = (y: number) => H / 2 - y * 60;
  const verdict = degree <= 1 ? "underfit" : testMSE < 0.12 ? "good fit" : "OVERFIT — memorising noise";

  useEffect(() => {
    const c = canvasRef.current!; const ctx = c.getContext("2d")!; ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = "#e6e1d6"; ctx.beginPath(); ctx.moveTo(PAD, H / 2); ctx.lineTo(W - PAD, H / 2); ctx.stroke();
    // true curve
    ctx.strokeStyle = "#2e9e6b"; ctx.setLineDash([6, 4]); ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i <= 200; i++) { const x = -3 + (i / 200) * 6; i ? ctx.lineTo(px(x), py(trueF(x))) : ctx.moveTo(px(x), py(trueF(x))); } ctx.stroke();
    ctx.setLineDash([]);
    // fitted curve
    ctx.strokeStyle = "#5b53e0"; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let i = 0; i <= 300; i++) { const x = -3 + (i / 300) * 6; const y = evalPoly(co, x); const yy = Math.max(-3, Math.min(3, y)); i ? ctx.lineTo(px(x), py(yy)) : ctx.moveTo(px(x), py(yy)); } ctx.stroke();
    // points
    ctx.fillStyle = "#1c1c2b"; for (const [x, y] of TRAIN) { ctx.beginPath(); ctx.arc(px(x), py(y), 4.5, 0, 7); ctx.fill(); }
  }, [co]);

  return (
    <VizCard>
      <canvas ref={canvasRef} className="viz-stage" width={W} height={H}
        style={{ width: "100%", height: "auto", aspectRatio: `${W}/${H}` }} />
      <Readout items={[
        ["degree", String(degree)],
        ["train MSE", trainMSE.toFixed(3)],
        ["test MSE", <span style={{ color: testMSE > 0.3 ? "var(--bad)" : "var(--good)" }}>{testMSE.toFixed(3)}</span>],
        ["verdict", <b style={{ color: verdict.startsWith("OVER") ? "var(--bad)" : verdict === "underfit" ? "var(--ink-soft)" : "var(--good)" }}>{verdict}</b>],
      ]} />
      <div className="viz-controls">
        <Slider label="polynomial degree" value={degree} min={1} max={12} step={1} onChange={(v) => setDegree(v)} fmt={(v) => String(v)} />
        <span style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
          Green dashed = the true pattern; violet = the model; black dots = noisy training data. At
          <b> degree 1</b> the model underfits; around <b>3</b> it matches the truth; crank it to
          <b> 10–12</b> and it threads every dot (train MSE → 0) while the curve goes wild between them and
          <b style={{ color: "var(--bad)" }}> test MSE explodes</b>. That gap <em>is</em> overfitting.</span>
      </div>
    </VizCard>
  );
}
