import { useState } from "react";
import { VizCard, Chips } from "../components/ui";

// Slide a 3x3 kernel over a 7x7 "image"; click an output cell to see its dot-product computation.
const N = 7, K = 3, M = N - K + 1; // input 7x7, output 5x5

const IMAGES: Record<string, number[][]> = {
  edge: Array.from({ length: N }, () => Array.from({ length: N }, (_, j) => (j >= 3 ? 1 : 0))),
  diagonal: Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (j >= i ? 1 : 0))),
  dot: Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => (Math.abs(i - 3) <= 1 && Math.abs(j - 3) <= 1 ? 1 : 0))),
};
const KERNELS: Record<string, number[][]> = {
  "vertical edge": [[1, 0, -1], [1, 0, -1], [1, 0, -1]],
  "horizontal edge": [[1, 1, 1], [0, 0, 0], [-1, -1, -1]],
  blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map((r) => r.map((v) => v / 9)),
  sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
};

const conv = (img: number[][], ker: number[][]) =>
  Array.from({ length: M }, (_, i) =>
    Array.from({ length: M }, (_, j) => {
      let s = 0; for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) s += img[i + a][j + b] * ker[a][b];
      return s;
    }));

function Cell({ v, hot, dim, onClick, neg }: { v: number; hot?: boolean; dim?: boolean; onClick?: () => void; neg?: boolean }) {
  let bg: string;
  if (neg) { const t = Math.max(-1, Math.min(1, v / 3)); bg = t >= 0 ? `rgba(37,99,235,${t})` : `rgba(224,83,61,${-t})`; }
  else bg = `rgba(28,28,43,${v})`;
  return (
    <div onClick={onClick} style={{
      width: 34, height: 34, display: "grid", placeItems: "center", borderRadius: 5, cursor: onClick ? "pointer" : "default",
      background: bg, color: Math.abs(v) > 0.5 ? "#fff" : "#1c1c2b", fontFamily: "var(--mono)", fontSize: ".72rem",
      outline: hot ? "3px solid #5b53e0" : "1px solid var(--line)", opacity: dim ? 0.25 : 1, transition: "opacity .15s",
    }}>{Number.isInteger(v) ? v : v.toFixed(2)}</div>
  );
}
const Grid = ({ children, cols }: { children: React.ReactNode; cols: number }) =>
  <div style={{ display: "grid", gridTemplateColumns: `repeat(${cols}, 34px)`, gap: 3 }}>{children}</div>;

export default function ConvolutionExplorer() {
  const [imgKey, setImgKey] = useState("edge");
  const [kerKey, setKerKey] = useState("vertical edge");
  const [sel, setSel] = useState<[number, number]>([2, 2]);
  const img = IMAGES[imgKey], ker = KERNELS[kerKey], out = conv(img, ker);
  const [oi, oj] = sel;
  const terms: string[] = [];
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) terms.push(`${img[oi + a][oj + b]}·${ker[a][b]}`);

  return (
    <VizCard>
      <div style={{ display: "flex", gap: 26, flexWrap: "wrap", padding: 20, alignItems: "flex-start" }}>
        <div>
          <div style={{ fontSize: ".75rem", color: "var(--ink-faint)", marginBottom: 6 }}>INPUT 7×7</div>
          <Grid cols={N}>
            {img.flatMap((row, i) => row.map((v, j) => {
              const inPatch = i >= oi && i < oi + K && j >= oj && j < oj + K;
              return <Cell key={`${i}-${j}`} v={v} hot={inPatch} dim={!inPatch} />;
            }))}
          </Grid>
        </div>
        <div>
          <div style={{ fontSize: ".75rem", color: "var(--ink-faint)", marginBottom: 6 }}>KERNEL 3×3</div>
          <Grid cols={K}>{ker.flat().map((v, i) => <Cell key={i} v={v} neg />)}</Grid>
          <div style={{ fontSize: "1.5rem", textAlign: "center", color: "var(--ink-faint)", margin: "30px 0" }}>→</div>
        </div>
        <div>
          <div style={{ fontSize: ".75rem", color: "var(--ink-faint)", marginBottom: 6 }}>FEATURE MAP 5×5 (click a cell)</div>
          <Grid cols={M}>
            {out.flatMap((row, i) => row.map((v, j) =>
              <Cell key={`${i}-${j}`} v={v} neg hot={i === oi && j === oj} onClick={() => setSel([i, j])} />))}
          </Grid>
        </div>
      </div>
      <div className="viz-readout">
        <span style={{ minWidth: "100%" }}>output[{oi},{oj}] = {terms.join(" + ")} = <b>{out[oi][oj].toFixed(2)}</b></span>
      </div>
      <div className="viz-controls">
        <div><div style={{ fontSize: ".8rem", color: "var(--ink-faint)", marginBottom: 4 }}>IMAGE</div>
          <Chips value={imgKey} onChange={setImgKey} options={Object.keys(IMAGES).map((k) => ({ id: k, label: k }))} /></div>
        <div><div style={{ fontSize: ".8rem", color: "var(--ink-faint)", marginBottom: 4 }}>KERNEL</div>
          <Chips value={kerKey} onChange={setKerKey} options={Object.keys(KERNELS).map((k) => ({ id: k, label: k }))} /></div>
        <span style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
          The kernel slides over every 3×3 patch; each output cell is the <b>dot product</b> of the kernel
          with the patch it covers. Click feature-map cells to see the computation. Notice the
          <b> vertical-edge</b> kernel lights up exactly where the image changes left-to-right — that's a
          learned feature detector, applied everywhere with the same 9 weights.</span>
      </div>
    </VizCard>
  );
}
