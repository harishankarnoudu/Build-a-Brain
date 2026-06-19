import { useEffect, useRef, useState, useCallback } from "react";
import { VizCard, Slider, Chips } from "../components/ui";

// Race SGD / Momentum / RMSProp / Adam on a 2-D loss surface (heatmap + paths).
const C = 380, DOM = 1.1;
type Surface = "bowl" | "ravine";
const COEFF: Record<Surface, [number, number]> = { bowl: [1, 1], ravine: [1, 12] };

const OPTS = [
  { id: "sgd", name: "SGD", color: "#8a8a9c" },
  { id: "momentum", name: "Momentum", color: "#e0533d" },
  { id: "rmsprop", name: "RMSProp", color: "#2e9e6b" },
  { id: "adam", name: "Adam", color: "#5b53e0" },
] as const;

type State = { x: number; y: number; v: number[]; s: number[]; m: number[]; t: number; dead?: boolean };

export default function OptimizerRace() {
  const [surface, setSurface] = useState<Surface>("ravine");
  const [lr, setLr] = useState(0.04);
  const [running, setRunning] = useState(false);
  const [iter, setIter] = useState(0);
  const states = useRef<Record<string, State>>({});
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const paths = useRef<Record<string, [number, number][]>>({});
  const raf = useRef<number | null>(null);

  const px = (x: number) => ((x + DOM) / (2 * DOM)) * C;
  const py = (y: number) => C - ((y + DOM) / (2 * DOM)) * C;
  const grad = (x: number, y: number) => { const [a, b] = COEFF[surface]; return [2 * a * x, 2 * b * y]; };
  const loss = (x: number, y: number) => { const [a, b] = COEFF[surface]; return a * x * x + b * y * y; };

  const reset = useCallback(() => {
    const init = (): State => ({ x: -0.9, y: 0.85, v: [0, 0], s: [0, 0], m: [0, 0], t: 0 });
    states.current = Object.fromEntries(OPTS.map((o) => [o.id, init()]));
    paths.current = Object.fromEntries(OPTS.map((o) => [o.id, [[-0.9, 0.85]]]));
    setIter(0);
  }, [surface]);
  useEffect(() => { reset(); }, [reset]);

  const drawBg = useCallback((ctx: CanvasRenderingContext2D) => {
    const cell = 4;
    let maxL = loss(-DOM, -DOM);
    for (let i = 0; i < C; i += cell) for (let j = 0; j < C; j += cell) {
      const x = (i / C) * 2 * DOM - DOM, y = DOM - (j / C) * 2 * DOM;
      const t = Math.min(1, loss(x, y) / maxL);
      const shade = Math.round(255 - t * 90);
      ctx.fillStyle = `rgb(${shade},${shade - 6},${shade - 18})`;
      ctx.fillRect(i, j, cell, cell);
    }
    // contour rings
    ctx.strokeStyle = "rgba(91,83,224,.18)";
    for (let r = 0.1; r < 1.2; r += 0.18) {
      ctx.beginPath();
      for (let a = 0; a <= 6.3; a += 0.1) {
        const [ca, cb] = COEFF[surface];
        const x = (r / Math.sqrt(ca)) * Math.cos(a), y = (r / Math.sqrt(cb)) * Math.sin(a);
        a === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y));
      }
      ctx.stroke();
    }
  }, [surface]);

  const draw = useCallback(() => {
    const c = canvasRef.current; if (!c) return; const ctx = c.getContext("2d")!;
    drawBg(ctx);
    ctx.beginPath(); ctx.arc(px(0), py(0), 5, 0, 7); ctx.fillStyle = "#111"; ctx.fill(); // minimum
    for (const o of OPTS) {
      const path = paths.current[o.id]; if (!path) continue;
      ctx.strokeStyle = o.color; ctx.lineWidth = 2; ctx.beginPath();
      path.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(px(x), py(y)) : ctx.lineTo(px(x), py(y))));
      ctx.stroke();
      const [lx, ly] = path[path.length - 1];
      ctx.beginPath(); ctx.arc(px(lx), py(ly), 4, 0, 7); ctx.fillStyle = o.color; ctx.fill();
    }
  }, [drawBg]);

  useEffect(() => {
    if (!running) { if (raf.current) cancelAnimationFrame(raf.current); return; }
    let it = iter;
    const tick = () => {
      for (const o of OPTS) {
        const st = states.current[o.id]; if (st.dead) continue;
        const g = grad(st.x, st.y); let dx = 0, dy = 0; st.t++;
        if (o.id === "sgd") { dx = lr * g[0]; dy = lr * g[1]; }
        else if (o.id === "momentum") { st.v = [0.9 * st.v[0] + g[0], 0.9 * st.v[1] + g[1]]; dx = lr * st.v[0]; dy = lr * st.v[1]; }
        else if (o.id === "rmsprop") { st.s = [0.9 * st.s[0] + 0.1 * g[0] ** 2, 0.9 * st.s[1] + 0.1 * g[1] ** 2]; dx = lr * g[0] / (Math.sqrt(st.s[0]) + 1e-8); dy = lr * g[1] / (Math.sqrt(st.s[1]) + 1e-8); }
        else { st.m = [0.9 * st.m[0] + 0.1 * g[0], 0.9 * st.m[1] + 0.1 * g[1]]; st.s = [0.999 * st.s[0] + 0.001 * g[0] ** 2, 0.999 * st.s[1] + 0.001 * g[1] ** 2];
          const mh = [st.m[0] / (1 - 0.9 ** st.t), st.m[1] / (1 - 0.9 ** st.t)], sh = [st.s[0] / (1 - 0.999 ** st.t), st.s[1] / (1 - 0.999 ** st.t)];
          dx = lr * mh[0] / (Math.sqrt(sh[0]) + 1e-8); dy = lr * mh[1] / (Math.sqrt(sh[1]) + 1e-8); }
        st.x -= dx; st.y -= dy;
        if (Math.abs(st.x) > 5 || Math.abs(st.y) > 5) st.dead = true;
        else paths.current[o.id].push([st.x, st.y]);
      }
      it++; setIter(it); draw();
      if (it < 400) raf.current = requestAnimationFrame(tick); else setRunning(false);
    };
    raf.current = requestAnimationFrame(tick);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [running, lr, draw]); // eslint-disable-line

  useEffect(() => { const t = setTimeout(draw, 30); return () => clearTimeout(t); }, [draw, surface]);

  return (
    <VizCard>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, padding: 16, alignItems: "start" }}>
        <canvas ref={canvasRef} width={C} height={C}
          style={{ width: "100%", maxWidth: C, border: "1px solid var(--line)", borderRadius: 10 }} />
        <div style={{ display: "grid", gap: 12 }}>
          <div>
            <div style={{ fontSize: ".8rem", color: "var(--ink-faint)", marginBottom: 4 }}>LOSS SURFACE</div>
            <Chips value={surface} onChange={setSurface} options={[
              { id: "bowl" as Surface, label: "round bowl" }, { id: "ravine" as Surface, label: "stretched ravine" }]} />
          </div>
          <div style={{ display: "grid", gap: 4 }}>
            {OPTS.map((o) => {
              const st = states.current[o.id];
              return <div key={o.id} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: ".85rem" }}>
                <span style={{ width: 14, height: 14, background: o.color, borderRadius: 3 }} />
                <b>{o.name}</b>
                <span style={{ marginLeft: "auto", fontFamily: "var(--mono)", color: "var(--ink-soft)" }}>
                  loss {st ? loss(st.x, st.y).toFixed(3) : "—"}</span>
              </div>;
            })}
          </div>
          <Slider label="learning rate" value={lr} min={0.005} max={0.2} step={0.005} onChange={(v) => { setLr(v); reset(); }} fmt={(v) => v.toFixed(3)} />
          <div className="ctl-row">
            <button className="btn" onClick={() => setRunning((r) => !r)}>{running ? "⏸ Pause" : "▶ Race"}</button>
            <button className="btn ghost" onClick={() => { setRunning(false); reset(); setTimeout(draw, 30); }}>↺ Reset</button>
            <span style={{ fontFamily: "var(--mono)", fontSize: ".8rem" }}>iter {iter}</span>
          </div>
          <span style={{ color: "var(--ink-soft)", fontSize: ".86rem" }}>
            On the <b>stretched ravine</b>, plain <b>SGD</b> zig-zags slowly while <b>Adam</b> and
            <b> RMSProp</b> plunge straight to the minimum (black dot) — which is why Adam trains nearly
            every modern model.</span>
        </div>
      </div>
    </VizCard>
  );
}
