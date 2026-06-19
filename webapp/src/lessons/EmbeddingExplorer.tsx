import { useState } from "react";
import { VizCard, Readout } from "../components/ui";

// A curated 2-D word space: click two words to see cosine similarity; run the famous analogy.
type W = { word: string; x: number; y: number; theme: "animal" | "royal" };
const WORDS: W[] = [
  { word: "cat", x: -3.0, y: -2.0, theme: "animal" },
  { word: "dog", x: -2.5, y: -2.6, theme: "animal" },
  { word: "kitten", x: -3.4, y: -1.4, theme: "animal" },
  { word: "pet", x: -2.0, y: -1.8, theme: "animal" },
  { word: "man", x: 1.0, y: 1.5, theme: "royal" },
  { word: "woman", x: 1.0, y: 3.0, theme: "royal" },
  { word: "king", x: 2.5, y: 1.5, theme: "royal" },
  { word: "queen", x: 2.5, y: 3.0, theme: "royal" },
];
const SIZE = 440, OX = 60, OY = SIZE - 60, SC = 50;
const sx = (x: number) => OX + x * SC;
const sy = (y: number) => OY - y * SC;
const cos = (a: W, b: W) => (a.x * b.x + a.y * b.y) / (Math.hypot(a.x, a.y) * Math.hypot(b.x, b.y) || 1);
const get = (w: string) => WORDS.find((p) => p.word === w)!;

export default function EmbeddingExplorer() {
  const [sel, setSel] = useState<string[]>(["cat", "dog"]);
  const [analogy, setAnalogy] = useState(false);

  const pick = (w: string) =>
    setSel((s) => (s.includes(w) ? s.filter((x) => x !== w) : [...s.slice(-1), w]));

  const a = sel[0] ? get(sel[0]) : null, b = sel[1] ? get(sel[1]) : null;
  const sim = a && b ? cos(a, b) : null;

  // analogy king - man + woman
  const king = get("king"), man = get("man"), woman = get("woman");
  const res = { x: king.x - man.x + woman.x, y: king.y - man.y + woman.y };

  return (
    <VizCard className="">
      <svg className="viz-stage" viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ height: 440 }}>
        <line x1={OX} y1={20} x2={OX} y2={OY} stroke="#d8d3c6" />
        <line x1={OX} y1={OY} x2={SIZE - 20} y2={OY} stroke="#d8d3c6" />
        {/* similarity link */}
        {a && b && <line x1={sx(a.x)} y1={sy(a.y)} x2={sx(b.x)} y2={sy(b.y)} stroke="#5b53e0" strokeWidth={2} strokeDasharray="5 4" />}
        {/* analogy arrows */}
        {analogy && <g>
          <line x1={sx(man.x)} y1={sy(man.y)} x2={sx(woman.x)} y2={sy(woman.y)} stroke="#2e9e6b" strokeWidth={2} />
          <line x1={sx(king.x)} y1={sy(king.y)} x2={sx(res.x)} y2={sy(res.y)} stroke="#2e9e6b" strokeWidth={2} strokeDasharray="4 3" />
          <circle cx={sx(res.x)} cy={sy(res.y)} r={9} fill="none" stroke="#e0533d" strokeWidth={2} />
          <text x={sx(res.x) + 12} y={sy(res.y)} fontSize="12" fill="#e0533d">king − man + woman</text>
        </g>}
        {WORDS.map((w) => {
          const on = sel.includes(w.word);
          return (
            <g key={w.word} cursor="pointer" onClick={() => pick(w.word)}>
              <circle cx={sx(w.x)} cy={sy(w.y)} r={on ? 9 : 6}
                fill={on ? "#5b53e0" : w.theme === "animal" ? "#2563eb" : "#e0533d"} opacity={0.85} />
              <text x={sx(w.x) + 11} y={sy(w.y) + 4} fontSize="14" fontWeight={on ? 700 : 400}
                fill="var(--ink)">{w.word}</text>
            </g>
          );
        })}
      </svg>
      <Readout items={[
        ["selected", sel.join(" , ") || "—"],
        ["cosine sim", sim === null ? "—" : <span style={{ color: sim > 0.5 ? "var(--good)" : "var(--ink)" }}>{sim.toFixed(3)}</span>],
      ]} />
      <div className="viz-controls">
        <span style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
          Click any two words to measure their <b>cosine similarity</b> — same-theme words
          (cat·dog) score high, cross-theme (cat·king) score low, purely from their positions.</span>
        <div className="ctl-row">
          <button className={"btn" + (analogy ? "" : " ghost")} onClick={() => setAnalogy((v) => !v)}>
            {analogy ? "Hide" : "Show"} analogy: king − man + woman ≈ queen</button>
        </div>
        <span style={{ color: "var(--ink-soft)", fontSize: ".88rem" }}>
          The green arrows are parallel — the "man→woman" direction equals "king→queen". That's why
          adding it to <b>king</b> lands almost exactly on <b>queen</b>: relationships become directions in space.</span>
      </div>
    </VizCard>
  );
}
