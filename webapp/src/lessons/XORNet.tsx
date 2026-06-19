import { useEffect, useRef, useState, useCallback } from "react";
import { MLP, makeData, type Point } from "../viz/nn";
import { VizCard, Slider } from "../components/ui";

// Train a 1-hidden-layer net on XOR; vary hidden-neuron count and see the boundary form.
const C = 360;
const toPix = (v: number) => ((v + 1) / 2) * C;

export default function XORNet() {
  const [hidden, setHidden] = useState(4);
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [acc, setAcc] = useState(0);
  const net = useRef<MLP | null>(null);
  const data = useRef<Point[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const raf = useRef<number | null>(null);

  const rebuild = useCallback(() => {
    const sizes = hidden > 0 ? [2, hidden, 1] : [2, 1];
    net.current = new MLP(sizes, "tanh");
    data.current = makeData("xor", 160, 0.05);
    setEpoch(0); setAcc(0);
  }, [hidden]);
  useEffect(() => { rebuild(); }, [rebuild]);

  const draw = useCallback(() => {
    const c = canvasRef.current, model = net.current; if (!c || !model) return;
    const ctx = c.getContext("2d")!; const cell = 6;
    for (let i = 0; i < C; i += cell) for (let j = 0; j < C; j += cell) {
      const mx = (i / C) * 2 - 1, my = -((j / C) * 2 - 1);
      const t = model.forward([mx, my]);
      ctx.fillStyle = `rgba(${Math.round(224 + t * (37 - 224))},${Math.round(83 + t * 16)},${Math.round(61 + t * 174)},0.42)`;
      ctx.fillRect(i, j, cell, cell);
    }
    // hidden-unit boundary lines (faint): w0*x + w1*y + b = 0
    if (hidden > 0) {
      const L0 = model.layers[0];
      ctx.strokeStyle = "rgba(40,40,70,.35)"; ctx.lineWidth = 1;
      for (let j = 0; j < L0.W.length; j++) {
        const [w0, w1] = L0.W[j], b = L0.b[j];
        if (Math.abs(w1) < 1e-6) continue;
        const y1 = -(w0 * -1 + b) / w1, y2 = -(w0 * 1 + b) / w1;
        ctx.beginPath(); ctx.moveTo(toPix(-1), C - toPix(y1)); ctx.lineTo(toPix(1), C - toPix(y2)); ctx.stroke();
      }
    }
    for (const pt of data.current) {
      ctx.beginPath(); ctx.arc(toPix(pt.x), C - toPix(pt.y), 4, 0, 7);
      ctx.fillStyle = pt.label === 1 ? "#2563eb" : "#e0533d"; ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,.4)"; ctx.stroke();
    }
  }, [hidden]);

  useEffect(() => {
    if (!running) { if (raf.current) cancelAnimationFrame(raf.current); return; }
    let e = epoch;
    const tick = () => {
      const model = net.current!, pts = data.current;
      const X = pts.map((p) => [p.x, p.y]), Y = pts.map((p) => p.label);
      for (let s = 0; s < 4; s++) model.trainStep(X, Y, 0.5);
      let correct = 0; for (const pt of pts) if (Math.round(model.forward([pt.x, pt.y])) === pt.label) correct++;
      e += 4; setEpoch(e); setAcc(correct / pts.length); draw();
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [running, draw]); // eslint-disable-line

  useEffect(() => { const t = setTimeout(draw, 30); return () => clearTimeout(t); }, [draw]);

  return (
    <VizCard>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, padding: 16, alignItems: "start" }}>
        <div>
          <canvas ref={canvasRef} width={C} height={C}
            style={{ width: "100%", maxWidth: C, border: "1px solid var(--line)", borderRadius: 10, background: "#fff" }} />
          <div style={{ fontFamily: "var(--mono)", fontSize: ".85rem", marginTop: 8 }}>
            epoch <b>{epoch}</b> &nbsp; accuracy <b style={{ color: acc > 0.95 ? "var(--good)" : "var(--accent)" }}>{(acc * 100).toFixed(0)}%</b>
          </div>
        </div>
        <div style={{ display: "grid", gap: 12 }}>
          <Slider label="hidden neurons" value={hidden} min={0} max={8} step={1} onChange={(v) => setHidden(v)} fmt={(v) => String(v)} />
          <div className="ctl-row">
            <button className="btn" onClick={() => setRunning((r) => !r)}>{running ? "⏸ Pause" : "▶ Train"}</button>
            <button className="btn ghost" onClick={() => { setRunning(false); rebuild(); setTimeout(draw, 30); }}>↺ Reset</button>
          </div>
          <span style={{ color: "var(--ink-soft)", fontSize: ".88rem" }}>
            XOR's classes sit on opposite diagonals — no straight line splits them. With <b>0 hidden
            neurons</b> (just logistic regression) training sticks near 50–75%. Bump it to <b>2+</b> and the
            net carves a curved boundary to ~100%. The faint grey lines are each hidden neuron's own
            boundary — the network <em>combines</em> them into the curve.</span>
        </div>
      </div>
    </VizCard>
  );
}
