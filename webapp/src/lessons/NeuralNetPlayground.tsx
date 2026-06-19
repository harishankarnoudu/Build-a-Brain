import { useEffect, useRef, useState, useCallback } from "react";
import { MLP, makeData, type Act, type Dataset, type Point } from "../viz/nn";
import { VizCard, Slider, Chips } from "../components/ui";
import { usePersistentState } from "../store";

const CANVAS = 380, GRID = 58;
const toPix = (v: number) => ((v + 1) / 2) * CANVAS;      // math [-1,1] -> pixel
const toMath = (p: number) => (p / CANVAS) * 2 - 1;

export default function NeuralNetPlayground() {
  const [dataset, setDataset] = usePersistentState<Dataset>("pg:dataset", "circle");
  const [act, setAct] = usePersistentState<Act>("pg:act", "tanh");
  const [hidden, setHidden] = usePersistentState<number[]>("pg:hidden", [4, 4]);
  const [lr, setLr] = usePersistentState("pg:lr", 0.3);
  const [noise, setNoise] = usePersistentState("pg:noise", 0.1);
  const [running, setRunning] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [loss, setLoss] = useState(NaN);

  const net = useRef<MLP | null>(null);
  const data = useRef<Point[]>([]);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const raf = useRef<number | null>(null);

  const rebuild = useCallback(() => {
    net.current = new MLP([2, ...hidden, 1], act);
    data.current = makeData(dataset, 160, noise);
    setEpoch(0); setLoss(NaN);
  }, [hidden, act, dataset, noise]);

  useEffect(() => { rebuild(); }, [rebuild]);

  const draw = useCallback(() => {
    const c = canvasRef.current, model = net.current; if (!c || !model) return;
    const ctx = c.getContext("2d")!;
    // decision boundary
    const cell = CANVAS / GRID;
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const mx = toMath((i + 0.5) * cell), my = -toMath((j + 0.5) * cell);
        const p = model.forward([mx, my]);            // 0..1
        // blue (class1) <-> red (class0)
        const t = p; // 0 red, 1 blue
        const r = Math.round(224 + t * (37 - 224));
        const g = Math.round(83 + t * (99 - 83));
        const b = Math.round(61 + t * (235 - 61));
        ctx.fillStyle = `rgba(${r},${g},${b},0.45)`;
        ctx.fillRect(i * cell, j * cell, cell + 1, cell + 1);
      }
    }
    // data points
    for (const pt of data.current) {
      ctx.beginPath();
      ctx.arc(toPix(pt.x), CANVAS - toPix(pt.y), 4, 0, 7);
      ctx.fillStyle = pt.label === 1 ? "#2563eb" : "#e0533d";
      ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(0,0,0,.4)"; ctx.stroke();
    }
  }, []);

  // training loop
  useEffect(() => {
    if (!running) { if (raf.current) cancelAnimationFrame(raf.current); return; }
    let e = epoch;
    const tick = () => {
      const model = net.current!; const pts = data.current;
      const X = pts.map((p) => [p.x, p.y]); const Y = pts.map((p) => p.label);
      let l = 0;
      for (let s = 0; s < 3; s++) l = model.trainStep(X, Y, lr);   // a few steps per frame
      e += 3; setEpoch(e); setLoss(l); draw();
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [running, lr, draw]);   // eslint-disable-line

  // redraw when architecture/data changes
  useEffect(() => { draw(); }, [draw, epoch === 0 ? hidden : null, act, dataset, noise]);
  useEffect(() => { const t = setTimeout(draw, 30); return () => clearTimeout(t); }, [draw, hidden, act, dataset, noise]);

  const setLayer = (idx: number, delta: number) =>
    setHidden((h) => { const n = [...h]; n[idx] = Math.max(1, Math.min(8, n[idx] + delta)); return n; });
  const addLayer = () => setHidden((h) => (h.length < 4 ? [...h, 4] : h));
  const removeLayer = () => setHidden((h) => (h.length > 1 ? h.slice(0, -1) : h));

  return (
    <VizCard>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, padding: 16, alignItems: "start" }}>
        <div>
          <canvas ref={canvasRef} width={CANVAS} height={CANVAS}
            style={{ width: "100%", maxWidth: CANVAS, border: "1px solid var(--line)", borderRadius: 10, background: "#fff" }} />
          <div style={{ fontFamily: "var(--mono)", fontSize: ".85rem", marginTop: 8 }}>
            epoch <b>{epoch}</b> &nbsp; loss <b style={{ color: "var(--accent)" }}>{isNaN(loss) ? "—" : loss.toFixed(3)}</b>
          </div>
        </div>

        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <div style={{ fontSize: ".8rem", color: "var(--ink-faint)", marginBottom: 4 }}>DATASET</div>
            <Chips value={dataset} onChange={setDataset} options={[
              { id: "circle" as Dataset, label: "Circle" }, { id: "xor" as Dataset, label: "XOR" },
              { id: "gauss" as Dataset, label: "Gaussian" }, { id: "spiral" as Dataset, label: "Spiral" }]} />
          </div>
          <div>
            <div style={{ fontSize: ".8rem", color: "var(--ink-faint)", marginBottom: 4 }}>ACTIVATION</div>
            <Chips value={act} onChange={setAct} options={[
              { id: "tanh" as Act, label: "tanh" }, { id: "relu" as Act, label: "ReLU" }, { id: "sigmoid" as Act, label: "sigmoid" }]} />
          </div>
          <div>
            <div style={{ fontSize: ".8rem", color: "var(--ink-faint)", marginBottom: 4 }}>
              HIDDEN LAYERS &nbsp;[{hidden.join(", ")}]
            </div>
            <div className="ctl-row">
              {hidden.map((n, i) => (
                <span key={i} style={{ display: "inline-flex", alignItems: "center", gap: 4, border: "1px solid var(--line)", borderRadius: 8, padding: "2px 4px" }}>
                  <button className="chip" onClick={() => setLayer(i, -1)}>−</button>
                  <b style={{ minWidth: 14, textAlign: "center" }}>{n}</b>
                  <button className="chip" onClick={() => setLayer(i, +1)}>+</button>
                </span>
              ))}
              <button className="chip" onClick={addLayer}>+ layer</button>
              <button className="chip" onClick={removeLayer}>− layer</button>
            </div>
          </div>
          <Slider label="learning rate" value={lr} min={0.01} max={1.5} step={0.01} onChange={setLr} />
          <Slider label="noise" value={noise} min={0} max={0.5} step={0.05} onChange={(v) => setNoise(v)} />
          <div className="ctl-row">
            <button className="btn" onClick={() => setRunning((r) => !r)}>{running ? "⏸ Pause" : "▶ Train"}</button>
            <button className="btn ghost" onClick={() => { setRunning(false); rebuild(); setTimeout(draw, 30); }}>↺ Reset</button>
          </div>
        </div>
      </div>
      <div className="viz-controls" style={{ color: "var(--ink-soft)", fontSize: ".9rem", borderTop: "1px solid var(--line)" }}>
        <span>Hit <b>Train</b> and watch the background <b>decision boundary</b> bend to separate
        <b style={{ color: "#2563eb" }}> blue</b> from <b style={{ color: "#e0533d" }}> red</b>. Try <b>Spiral</b> with
        one tiny layer (it can't) then add neurons/layers (it can) — that's the power of depth. Switch
        <b> ReLU vs tanh</b>, and push the learning rate too high to watch it destabilise.</span>
      </div>
    </VizCard>
  );
}
