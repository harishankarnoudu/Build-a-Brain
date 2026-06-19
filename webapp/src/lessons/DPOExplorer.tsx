import { useState } from "react";
import { VizCard, Slider, Readout } from "../components/ui";

// DPO / Bradley-Terry preference loss:  loss = -ln σ( β·(Δ_w - Δ_l) ).
const W = 560, H = 300, PAD = 40;
const sig = (z: number) => 1 / (1 + Math.exp(-z));

export default function DPOExplorer() {
  const [dw, setDw] = useState(0.5);   // preferred answer's log-prob ratio vs reference
  const [dl, setDl] = useState(-0.3);  // rejected answer's log-prob ratio
  const [beta, setBeta] = useState(1.0);

  const margin = beta * (dw - dl);
  const pPrefer = sig(margin);
  const loss = -Math.log(Math.max(1e-9, pPrefer));

  const mMin = -5, mMax = 5, lMax = 5;
  const mx = (m: number) => PAD + ((m - mMin) / (mMax - mMin)) * (W - 2 * PAD);
  const ly = (l: number) => H - PAD - (Math.min(l, lMax) / lMax) * (H - 2 * PAD);
  const curve: string[] = [];
  for (let i = 0; i <= 200; i++) { const m = mMin + (i / 200) * (mMax - mMin); curve.push(`${i ? "L" : "M"}${mx(m)},${ly(-Math.log(Math.max(1e-9, sig(m))))}`); }

  return (
    <VizCard>
      <svg className="viz-stage" viewBox={`0 0 ${W} ${H}`} style={{ height: 300 }}>
        <line x1={PAD} y1={H - PAD} x2={W - PAD} y2={H - PAD} stroke="#b9b3a5" />
        <line x1={mx(0)} y1={PAD} x2={mx(0)} y2={H - PAD} stroke="#e6e1d6" />
        <path d={curve.join(" ")} fill="none" stroke="#5b53e0" strokeWidth={2.5} />
        <circle cx={mx(margin)} cy={ly(loss)} r={7} fill="#e0533d" />
        <text x={W - PAD - 4} y={H - PAD + 18} textAnchor="end" fontSize="11" fill="var(--ink-faint)">margin = β·(Δw − Δl) →</text>
        <text x={PAD} y={PAD - 14} fontSize="11" fill="var(--ink-faint)">loss = −ln σ(margin)</text>
        <text x={mx(margin) + 10} y={ly(loss) - 8} fontSize="12" fill="#e0533d" fontWeight="700">loss {loss.toFixed(2)}</text>
      </svg>
      <Readout items={[
        ["margin", margin.toFixed(2)],
        ["P(prefer winner)", `${(pPrefer * 100).toFixed(0)}%`],
        ["DPO loss", <span style={{ color: loss < 0.4 ? "var(--good)" : loss > 1 ? "var(--bad)" : "var(--ink)" }}>{loss.toFixed(3)}</span>],
      ]} />
      <div className="viz-controls">
        <Slider label="Δ winner" value={dw} min={-2} max={2} step={0.1} onChange={setDw} />
        <Slider label="Δ loser" value={dl} min={-2} max={2} step={0.1} onChange={setDl} />
        <Slider label="β (KL strength)" value={beta} min={0.1} max={3} step={0.1} onChange={setBeta} />
        <span style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
          <b>Δ</b> is how much more (or less) likely the model makes an answer versus the frozen reference.
          When the <b>winner's Δ exceeds the loser's</b>, the margin is positive, <code>P(prefer winner)</code>
          rises above 50%, and the loss drops toward 0. Training pushes the winner up and the loser down —
          aligning the model to human preference with a single, stable loss (no reward model, no RL).</span>
      </div>
    </VizCard>
  );
}
