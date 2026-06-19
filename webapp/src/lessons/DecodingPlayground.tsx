import { useState } from "react";
import { VizCard, Slider, Chips } from "../components/ui";

// Show how temperature / top-k / top-p reshape a next-token distribution, then sample from it.
const TOKENS = ["the", "cat", "sat", "on", "mat", "ran", "dog", "fast"];
const BASE = [2.4, 2.0, 1.2, 0.9, 0.4, 0.1, -0.3, -0.8]; // raw logits

function softmax(z: number[]) {
  const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

export default function DecodingPlayground() {
  const [temp, setTemp] = useState(1.0);
  const [mode, setMode] = useState<"none" | "topk" | "topp">("none");
  const [k, setK] = useState(3);
  const [p, setP] = useState(0.9);
  const [sampled, setSampled] = useState<number | null>(null);

  // 1) temperature
  let probs = softmax(BASE.map((l) => l / Math.max(0.05, temp)));
  // 2) optional truncation
  const order = probs.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]);
  const keep = new Set<number>();
  if (mode === "topk") {
    order.slice(0, k).forEach(([, i]) => keep.add(i));
  } else if (mode === "topp") {
    let cum = 0;
    for (const [v, i] of order) { keep.add(i); cum += v; if (cum >= p) break; }
  } else {
    order.forEach(([, i]) => keep.add(i));
  }
  const masked = probs.map((v, i) => (keep.has(i) ? v : 0));
  const z = masked.reduce((a, b) => a + b, 0);
  probs = masked.map((v) => v / z);
  const entropy = -probs.reduce((s, q) => (q > 0 ? s + q * Math.log(q) : s), 0);

  function sample() {
    const r = Math.random(); let c = 0;
    for (let i = 0; i < probs.length; i++) { c += probs[i]; if (r <= c) { setSampled(i); return; } }
  }

  const maxP = Math.max(...probs, 0.001);
  return (
    <VizCard>
      <div style={{ padding: "20px 18px 8px" }}>
        {TOKENS.map((t, i) => (
          <div key={t} style={{ display: "grid", gridTemplateColumns: "60px 1fr 56px", alignItems: "center", gap: 10, margin: "4px 0" }}>
            <code style={{ textAlign: "right", background: "none", color: sampled === i ? "var(--accent)" : "var(--ink)", fontWeight: sampled === i ? 700 : 400 }}>{t}</code>
            <div style={{ background: "var(--paper-2)", borderRadius: 6, height: 20, overflow: "hidden" }}>
              <div style={{ width: `${(probs[i] / maxP) * 100}%`, height: "100%", borderRadius: 6,
                background: probs[i] === 0 ? "#ccc" : (sampled === i ? "var(--accent-2)" : "var(--accent)"),
                transition: "width .25s ease" }} />
            </div>
            <span style={{ fontFamily: "var(--mono)", fontSize: ".8rem", color: probs[i] === 0 ? "var(--ink-faint)" : "var(--ink)" }}>
              {(probs[i] * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <div className="viz-readout">
        <span>strategy <b>{mode === "none" ? "pure sampling" : mode === "topk" ? `top-k (k=${k})` : `top-p (p=${p})`}</b></span>
        <span>temperature <b>{temp.toFixed(2)}</b></span>
        <span>entropy <b>{entropy.toFixed(2)}</b></span>
        {sampled !== null && <span>sampled → <b style={{ color: "var(--accent-2)" }}>{TOKENS[sampled]}</b></span>}
      </div>
      <div className="viz-controls">
        <Slider label="temperature" value={temp} min={0.1} max={2} step={0.05} onChange={setTemp} />
        <Chips value={mode} onChange={(m) => setMode(m)} options={[
          { id: "none" as const, label: "pure sampling" }, { id: "topk" as const, label: "top-k" }, { id: "topp" as const, label: "top-p" }]} />
        {mode === "topk" && <Slider label="k" value={k} min={1} max={8} step={1} onChange={(v) => setK(v)} fmt={(v) => String(v)} />}
        {mode === "topp" && <Slider label="p" value={p} min={0.1} max={1} step={0.05} onChange={setP} />}
        <div className="ctl-row"><button className="btn" onClick={sample}>🎲 Sample a token</button></div>
        <span style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
          <b>Low temperature</b> (→0) sharpens toward greedy "the"; <b>high</b> (→2) flattens to near-random.
          <b> top-k</b> keeps only the k tallest bars; <b>top-p</b> keeps the smallest set covering p of the
          probability. Sample repeatedly to feel how the strategy changes what comes out.</span>
      </div>
    </VizCard>
  );
}
