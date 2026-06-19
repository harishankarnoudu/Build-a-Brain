import { useState } from "react";
import { VizCard, Slider } from "../components/ui";
import { M } from "../components/Math";

// Step through forward + backward on the scalar net  z=wx+b ; a=σ(z) ; L=(a-y)^2.
const sig = (z: number) => 1 / (1 + Math.exp(-z));
const W = 580, H = 300;

type Node = { id: string; x: number; y: number; label: string };
const NODES: Node[] = [
  { id: "x", x: 50, y: 70, label: "x" },
  { id: "w", x: 50, y: 150, label: "w" },
  { id: "b", x: 50, y: 230, label: "b" },
  { id: "z", x: 230, y: 150, label: "z" },
  { id: "y", x: 360, y: 55, label: "y" },
  { id: "a", x: 360, y: 150, label: "a" },
  { id: "L", x: 500, y: 150, label: "L" },
];
const EDGES: [string, string][] = [["x", "z"], ["w", "z"], ["b", "z"], ["z", "a"], ["a", "L"], ["y", "L"]];

const STEPS = [
  { phase: "forward", hot: ["z"], title: "Forward · compute z", tex: "z = w\\cdot x + b" },
  { phase: "forward", hot: ["a"], title: "Forward · activate", tex: "a = \\sigma(z)" },
  { phase: "forward", hot: ["L"], title: "Forward · loss", tex: "L = (a - y)^2" },
  { phase: "backward", hot: ["L", "a"], title: "Backward · ∂L/∂a", tex: "\\frac{\\partial L}{\\partial a} = 2(a - y)" },
  { phase: "backward", hot: ["a", "z"], title: "Backward · ∂L/∂z (chain rule)", tex: "\\frac{\\partial L}{\\partial z} = \\frac{\\partial L}{\\partial a}\\cdot a(1-a)" },
  { phase: "backward", hot: ["z", "w", "b"], title: "Backward · ∂L/∂w, ∂L/∂b", tex: "\\frac{\\partial L}{\\partial w} = \\frac{\\partial L}{\\partial z}\\cdot x,\\quad \\frac{\\partial L}{\\partial b} = \\frac{\\partial L}{\\partial z}" },
];

export default function BackpropWalkthrough() {
  const [w, setW] = useState(0.5);
  const [b, setB] = useState(0.0);
  const [x, setX] = useState(1.0);
  const [y, setY] = useState(1.0);
  const [step, setStep] = useState(5);

  // forward
  const z = w * x + b, a = sig(z), L = (a - y) ** 2;
  // backward
  const dLda = 2 * (a - y), dLdz = dLda * a * (1 - a);
  const grads: Record<string, number> = { L: 1, a: dLda, z: dLdz, w: dLdz * x, b: dLdz, x: dLdz * w, y: -2 * (a - y) };
  const vals: Record<string, number> = { x, w, b, z, a, y, L };

  const cur = STEPS[step];
  const showGrad = cur.phase === "backward";
  const pos = (id: string) => NODES.find((n) => n.id === id)!;

  function trainStep() { setW((ww) => ww - 0.5 * (dLdz * x)); setB((bb) => bb - 0.5 * dLdz); }

  return (
    <VizCard className="dark">
      <svg className="viz-stage" viewBox={`0 0 ${W} ${H}`} style={{ height: 300 }}>
        {EDGES.map(([f, t], i) => {
          const A = pos(f), B = pos(t);
          const hot = cur.hot.includes(f) && cur.hot.includes(t);
          return <line key={i} x1={A.x + 22} y1={A.y} x2={B.x - 22} y2={B.y}
            stroke={hot ? (showGrad ? "#e0533d" : "#9d94ff") : "#3a3c5e"} strokeWidth={hot ? 3 : 1.5}
            markerEnd={showGrad ? "url(#gb)" : "url(#gf)"} />;
        })}
        <defs>
          <marker id="gf" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#9d94ff" /></marker>
          <marker id="gb" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto"><path d="M6,0 L0,3 L6,6 Z" fill="#e0533d" /></marker>
        </defs>
        {NODES.map((n) => {
          const hot = cur.hot.includes(n.id);
          return (
            <g key={n.id}>
              <circle cx={n.x} cy={n.y} r={22} fill={hot ? "#2a2c55" : "#1d1f38"}
                stroke={hot ? (showGrad ? "#e0533d" : "#9d94ff") : "#3a3c5e"} strokeWidth={hot ? 3 : 1.5} />
              <text x={n.x} y={n.y - 1} textAnchor="middle" fontSize="15" fontWeight="700" fill="#e8e8f0">{n.label}</text>
              <text x={n.x} y={n.y + 13} textAnchor="middle" fontSize="9" fill="#8fd0c8" fontFamily="monospace">{vals[n.id].toFixed(2)}</text>
              {showGrad && grads[n.id] !== undefined &&
                <text x={n.x} y={n.y + 38} textAnchor="middle" fontSize="10" fill="#e0533d" fontFamily="monospace">∂={grads[n.id].toFixed(2)}</text>}
            </g>
          );
        })}
        <text x={20} y={24} fontSize="12" fill={showGrad ? "#e0533d" : "#9d94ff"} fontWeight="700">
          {showGrad ? "◄ BACKWARD (gradients)" : "FORWARD (values) ►"}
        </text>
      </svg>

      <div className="viz-readout">
        <span style={{ minWidth: "100%", color: "#e8e8f0" }}><b style={{ color: "#9d94ff" }}>{cur.title}:</b>{" "}
          <M>{cur.tex}</M></span>
        <span>z=<b>{z.toFixed(3)}</b></span><span>a=<b>{a.toFixed(3)}</b></span>
        <span>L=<b>{L.toFixed(3)}</b></span>
        {showGrad && <><span>∂L/∂w=<b style={{ color: "#e0533d" }}>{(dLdz * x).toFixed(3)}</b></span>
          <span>∂L/∂b=<b style={{ color: "#e0533d" }}>{dLdz.toFixed(3)}</b></span></>}
      </div>

      <div className="viz-controls">
        <div className="ctl-row">
          <button className="btn ghost" onClick={() => setStep((s) => Math.max(0, s - 1))} disabled={step === 0}>◄ Prev</button>
          <button className="btn" onClick={() => setStep((s) => Math.min(STEPS.length - 1, s + 1))} disabled={step === STEPS.length - 1}>Next step ►</button>
          <button className="btn ghost" onClick={trainStep}>↓ Take a gradient step (lr=0.5)</button>
        </div>
        <Slider label="w" value={w} min={-3} max={3} step={0.1} onChange={setW} />
        <Slider label="b" value={b} min={-3} max={3} step={0.1} onChange={setB} />
        <Slider label="x (input)" value={x} min={-3} max={3} step={0.1} onChange={setX} />
        <Slider label="y (target)" value={y} min={0} max={1} step={0.1} onChange={setY} />
        <span style={{ color: "#c8c8e0", fontSize: ".88rem" }}>
          Walk Forward (blue, left→right) to build the loss, then Backward (red, right→left) where each
          node multiplies its <b>local gradient</b> by the one handed back — the chain rule. Hit
          <b> "Take a gradient step"</b> a few times and watch <b>L</b> shrink as the network learns.</span>
      </div>
    </VizCard>
  );
}
