import { useEffect, useRef, useState } from "react";
import { VizCard, Readout, Slider } from "../components/ui";
import { usePersistentState } from "../store";

// Roll a ball down the loss bowl L(w) = (w-3)^2 with gradient descent. Animate steps; tune lr.
const SIZE_W = 640, SIZE_H = 360, PAD = 40;
const L = (w: number) => (w - 3) ** 2;
const dL = (w: number) => 2 * (w - 3);
const WMIN = -2, WMAX = 8, LMAX = 60;

export default function GradientDescent() {
  const [lr, setLr] = usePersistentState("gd2d:lr", 0.1);
  const [path, setPath] = useState<number[]>([0]);
  const [running, setRunning] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timer = useRef<number | null>(null);

  const wx = (w: number) => PAD + ((w - WMIN) / (WMAX - WMIN)) * (SIZE_W - 2 * PAD);
  const ly = (l: number) => SIZE_H - PAD - (Math.min(l, LMAX) / LMAX) * (SIZE_H - 2 * PAD);

  function step() {
    setPath((p) => {
      const w = p[p.length - 1];
      const next = w - lr * dL(w);
      if (!isFinite(next) || Math.abs(next) > 1e6) { setRunning(false); return [...p, Math.sign(next) * 1e6]; }
      return [...p, next];
    });
  }
  function reset() { setRunning(false); setPath([0]); }

  useEffect(() => {
    if (!running) { if (timer.current) clearInterval(timer.current); return; }
    timer.current = window.setInterval(step, 220);
    return () => { if (timer.current) clearInterval(timer.current); };
  }, [running, lr]);

  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext("2d")!;
    ctx.clearRect(0, 0, SIZE_W, SIZE_H);
    // axes
    ctx.strokeStyle = "#b9b3a5"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD, SIZE_H - PAD); ctx.lineTo(SIZE_W - PAD, SIZE_H - PAD); ctx.stroke();
    // loss curve
    ctx.strokeStyle = "#2563eb"; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const w = WMIN + (i / 200) * (WMAX - WMIN);
      i === 0 ? ctx.moveTo(wx(w), ly(L(w))) : ctx.lineTo(wx(w), ly(L(w)));
    }
    ctx.stroke();
    // minimum
    ctx.strokeStyle = "#2e9e6b"; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(wx(3), PAD); ctx.lineTo(wx(3), SIZE_H - PAD); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#2e9e6b"; ctx.font = "13px monospace"; ctx.fillText("minimum w=3", wx(3) + 6, PAD + 14);
    // path
    ctx.strokeStyle = "#e0533d"; ctx.fillStyle = "#e0533d"; ctx.lineWidth = 1.5;
    path.forEach((w, i) => {
      const cw = Math.max(WMIN, Math.min(WMAX, w));
      const X = wx(cw), Y = ly(L(w));
      if (i > 0) {
        const pw = Math.max(WMIN, Math.min(WMAX, path[i - 1]));
        ctx.beginPath(); ctx.moveTo(wx(pw), ly(L(path[i - 1]))); ctx.lineTo(X, Y); ctx.stroke();
      }
      ctx.beginPath(); ctx.arc(X, Y, i === path.length - 1 ? 7 : 4, 0, 7); ctx.fill();
    });
  }, [path]);

  const w = path[path.length - 1];
  const diverged = Math.abs(w) > 1e5;

  return (
    <VizCard>
      <canvas ref={canvasRef} className="viz-stage" width={SIZE_W} height={SIZE_H}
        style={{ width: "100%", height: "auto", aspectRatio: `${SIZE_W}/${SIZE_H}` }} />
      <Readout items={[
        ["step", String(path.length - 1)],
        ["w =", diverged ? <span style={{ color: "var(--bad)" }}>diverged!</span> : w.toFixed(3)],
        ["loss =", diverged ? "∞" : L(w).toFixed(3)],
        ["learning rate", lr.toFixed(2)],
      ]} />
      <div className="viz-controls">
        <div className="ctl-row">
          <button className="btn" onClick={() => setRunning((r) => !r)}>{running ? "⏸ Pause" : "▶ Run"}</button>
          <button className="btn ghost" onClick={step} disabled={running}>Step</button>
          <button className="btn ghost" onClick={reset}>↺ Reset</button>
        </div>
        <Slider label="learning rate" value={lr} min={0.01} max={1.1} step={0.01} onChange={(v) => { setLr(v); reset(); }} />
        <span style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
          Try <code>0.1</code> (smooth), <code>0.01</code> (slow crawl), <code>1.0</code> (overshoots/bounces),
          and <code>1.05</code> (<b style={{ color: "var(--bad)" }}>diverges</b> — flies uphill). This one slider
          is why "learning rate too high" is the #1 training failure.
        </span>
      </div>
    </VizCard>
  );
}
