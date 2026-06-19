import { useEffect, useRef, useState, useCallback } from "react";
import { makeData, type Dataset, type Point } from "../viz/nn";
import { VizCard, Slider, Chips } from "../components/ui";

const C = 360;
const toPix = (v: number) => ((v + 1) / 2) * C;

export default function LogisticBoundary() {
  const [dataset, setDataset] = useState<Dataset>("gauss");
  const [lr, setLr] = useState(0.5);
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [acc, setAcc] = useState(0);

  const w = useRef<[number, number]>([0, 0]);
  const b = useRef(0);
  const data = useRef<Point[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const raf = useRef<number | null>(null);

  const sig = (z: number) => 1 / (1 + Math.exp(-z));

  const rebuild = useCallback(() => {
    w.current = [Math.random() * 0.2, Math.random() * 0.2]; b.current = 0;
    data.current = makeData(dataset, 160, 0.08); setEpoch(0); setAcc(0);
  }, [dataset]);
  useEffect(() => { rebuild(); }, [rebuild]);

  const draw = useCallback(() => {
    const c = canvasRef.current; if (!c) return; const ctx = c.getContext("2d")!;
    const [w0, w1] = w.current, bb = b.current; const cell = 6;
    for (let i = 0; i < C; i += cell) for (let j = 0; j < C; j += cell) {
      const mx = (i / C) * 2 - 1, my = -((j / C) * 2 - 1);
      const t = sig(w0 * mx + w1 * my + bb);
      ctx.fillStyle = `rgba(${Math.round(224 + t * (37 - 224))},${Math.round(83 + t * (99 - 83))},${Math.round(61 + t * (235 - 61))},0.4)`;
      ctx.fillRect(i, j, cell, cell);
    }
    for (const pt of data.current) {
      ctx.beginPath(); ctx.arc(toPix(pt.x), C - toPix(pt.y), 4, 0, 7);
      ctx.fillStyle = pt.label === 1 ? "#2563eb" : "#e0533d"; ctx.fill();
      ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,.4)"; ctx.stroke();
    }
  }, []);

  useEffect(() => {
    if (!running) { if (raf.current) cancelAnimationFrame(raf.current); return; }
    let e = epoch;
    const tick = () => {
      const pts = data.current; let gw0 = 0, gw1 = 0, gb = 0, correct = 0;
      for (const pt of pts) {
        const p = sig(w.current[0] * pt.x + w.current[1] * pt.y + b.current);
        const err = p - pt.label; gw0 += err * pt.x; gw1 += err * pt.y; gb += err;
        if (Math.round(p) === pt.label) correct++;
      }
      const n = pts.length;
      w.current = [w.current[0] - lr * gw0 / n, w.current[1] - lr * gw1 / n];
      b.current -= lr * gb / n;
      e += 1; setEpoch(e); setAcc(correct / n); draw();
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [running, lr, draw]); // eslint-disable-line

  useEffect(() => { const t = setTimeout(draw, 30); return () => clearTimeout(t); }, [draw, dataset]);

  return (
    <VizCard>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, padding: 16, alignItems: "start" }}>
        <div>
          <canvas ref={canvasRef} width={C} height={C}
            style={{ width: "100%", maxWidth: C, border: "1px solid var(--line)", borderRadius: 10, background: "#fff" }} />
          <div style={{ fontFamily: "var(--mono)", fontSize: ".85rem", marginTop: 8 }}>
            epoch <b>{epoch}</b> &nbsp; accuracy <b style={{ color: "var(--accent)" }}>{(acc * 100).toFixed(0)}%</b>
          </div>
        </div>
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <div style={{ fontSize: ".8rem", color: "var(--ink-faint)", marginBottom: 4 }}>DATASET</div>
            <Chips value={dataset} onChange={setDataset} options={[
              { id: "gauss" as Dataset, label: "Gaussian (separable)" },
              { id: "circle" as Dataset, label: "Circle" },
              { id: "xor" as Dataset, label: "XOR" }]} />
          </div>
          <Slider label="learning rate" value={lr} min={0.05} max={2} step={0.05} onChange={setLr} />
          <div className="ctl-row">
            <button className="btn" onClick={() => setRunning((r) => !r)}>{running ? "⏸ Pause" : "▶ Train"}</button>
            <button className="btn ghost" onClick={() => { setRunning(false); rebuild(); setTimeout(draw, 30); }}>↺ Reset</button>
          </div>
          <span style={{ color: "var(--ink-soft)", fontSize: ".88rem" }}>
            Logistic regression can only draw a <b>straight</b> boundary. It nails the
            <b> Gaussian</b> blobs (~100%) but is stuck near 50–75% on <b>Circle</b>/<b>XOR</b> — no line
            separates them. That failure is exactly why we need the hidden layers of a neural network.</span>
        </div>
      </div>
    </VizCard>
  );
}
