import { useState } from "react";
import { VizCard, Slider, Readout } from "../components/ui";

// Drag a 3-outcome distribution and watch entropy peak when uniform, vanish when certain.
export default function EntropyExplorer() {
  const [raw, setRaw] = useState([0.5, 0.3, 0.2]);
  const sum = raw.reduce((a, b) => a + b, 0);
  const p = raw.map((v) => v / sum);
  const H = -p.reduce((s, q) => (q > 0 ? s + q * Math.log(q) : s), 0);
  const Hmax = Math.log(3);
  const labels = ["A", "B", "C"];
  const colors = ["#2563eb", "#2e9e6b", "#e0533d"];

  return (
    <VizCard>
      <div style={{ padding: "22px 24px 10px", display: "flex", gap: 28, alignItems: "flex-end", height: 200 }}>
        {p.map((q, i) => (
          <div key={i} style={{ flex: 1, textAlign: "center" }}>
            <div style={{ fontFamily: "var(--mono)", fontSize: ".8rem", marginBottom: 4 }}>{(q * 100).toFixed(0)}%</div>
            <div style={{ height: 150, display: "flex", alignItems: "flex-end" }}>
              <div style={{ width: "100%", height: `${q * 100}%`, background: colors[i], borderRadius: "6px 6px 0 0", transition: "height .15s" }} />
            </div>
            <div style={{ fontWeight: 700, marginTop: 6 }}>{labels[i]}</div>
          </div>
        ))}
        {/* entropy gauge */}
        <div style={{ width: 90, textAlign: "center" }}>
          <div style={{ fontSize: ".75rem", color: "var(--ink-faint)" }}>ENTROPY</div>
          <div style={{ fontSize: "1.6rem", fontWeight: 700, color: "var(--accent)", fontFamily: "var(--serif)" }}>{H.toFixed(2)}</div>
          <div style={{ height: 10, background: "var(--paper-2)", borderRadius: 6, overflow: "hidden", marginTop: 4 }}>
            <div style={{ width: `${(H / Hmax) * 100}%`, height: "100%", background: "var(--accent)" }} />
          </div>
          <div style={{ fontSize: ".7rem", color: "var(--ink-faint)" }}>max ln3 = 1.10</div>
        </div>
      </div>
      <Readout items={[
        ["H =", `${H.toFixed(3)} nats`],
        ["state", H > 1.05 ? "max uncertainty" : H < 0.4 ? "nearly certain" : "in between"],
      ]} />
      <div className="viz-controls">
        {[0, 1, 2].map((i) => (
          <Slider key={i} label={labels[i]} value={raw[i]} min={0.01} max={1} step={0.01}
            onChange={(v) => setRaw((r) => r.map((x, j) => (j === i ? v : x)))} fmt={(v) => v.toFixed(2)} />
        ))}
        <span style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
          Make the three bars <b>equal</b> → entropy peaks at <code>ln 3 ≈ 1.10</code> (you can't predict the
          outcome). Push one bar near <b>100%</b> → entropy collapses toward 0 (no surprise). This is the
          quantity a model's cross-entropy loss is built from.</span>
      </div>
    </VizCard>
  );
}
