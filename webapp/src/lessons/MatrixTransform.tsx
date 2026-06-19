import { VizCard, Readout, Slider } from "../components/ui";
import { usePersistentState } from "../store";

// Visualise a 2x2 matrix W as a transformation of the plane (3Blue1Brown style):
// transformed grid + unit square + basis vectors î, ĵ, and a tracked input vector x -> Wx.
const SIZE = 420, ORIGIN = SIZE / 2, SCALE = 55;
const S = (x: number, y: number) => [ORIGIN + x * SCALE, ORIGIN - y * SCALE] as const;

const PRESETS: Record<string, number[]> = {
  identity: [1, 0, 0, 1],
  scale: [2, 0, 0, 2],
  shear: [1, 1, 0, 1],
  rotate: [0, -1, 1, 0],
  flip: [-1, 0, 0, 1],
};

export default function MatrixTransform() {
  const [W, setW] = usePersistentState("mat2d:W", [1, 1.2, 0.4, 1]); // [w00,w01,w10,w11]
  const [x, setX] = usePersistentState("mat2d:x", { x: 1, y: 1 });
  const [w00, w01, w10, w11] = W;
  const apply = (px: number, py: number) => [w00 * px + w01 * py, w10 * px + w11 * py] as const;
  const Wx = apply(x.x, x.y);
  const det = w00 * w11 - w01 * w10;

  const lines: string[] = [];
  // transformed grid
  const gridEls: React.ReactNode[] = [];
  for (let g = -5; g <= 5; g++) {
    const [a1x, a1y] = apply(g, -5), [b1x, b1y] = apply(g, 5);
    const [c1x, c1y] = apply(-5, g), [d1x, d1y] = apply(5, g);
    const [sa, sb] = [S(a1x, a1y), S(b1x, b1y)];
    const [sc, sd] = [S(c1x, c1y), S(d1x, d1y)];
    const main = g === 0;
    gridEls.push(
      <line key={`v${g}`} x1={sa[0]} y1={sa[1]} x2={sb[0]} y2={sb[1]} stroke={main ? "#b9b3a5" : "#e6e1d6"} />,
      <line key={`h${g}`} x1={sc[0]} y1={sc[1]} x2={sd[0]} y2={sd[1]} stroke={main ? "#b9b3a5" : "#e6e1d6"} />
    );
  }
  void lines;

  // unit square image
  const sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map(([px, py]) => { const [tx, ty] = apply(px, py); return S(tx, ty); });
  const sqPath = sq.map((p, i) => `${i === 0 ? "M" : "L"}${p[0]},${p[1]}`).join(" ") + " Z";

  const ihat = S(...apply(1, 0)), jhat = S(...apply(0, 1));
  const xs = S(x.x, x.y), wxs = S(...Wx);

  return (
    <VizCard>
      <svg className="viz-stage" viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ height: 420 }}>
        <defs>
          <marker id="ih" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#2e9e6b" /></marker>
          <marker id="jh" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#e0533d" /></marker>
          <marker id="xm" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="#5b53e0" /></marker>
        </defs>
        {gridEls}
        <path d={sqPath} fill="#5b53e0" opacity={0.13} stroke="#5b53e0" strokeWidth={1.5} />
        {/* basis vectors */}
        <line x1={ORIGIN} y1={ORIGIN} x2={ihat[0]} y2={ihat[1]} stroke="#2e9e6b" strokeWidth={3} markerEnd="url(#ih)" />
        <line x1={ORIGIN} y1={ORIGIN} x2={jhat[0]} y2={jhat[1]} stroke="#e0533d" strokeWidth={3} markerEnd="url(#jh)" />
        {/* tracked vector */}
        <line x1={ORIGIN} y1={ORIGIN} x2={xs[0]} y2={xs[1]} stroke="#bbb" strokeWidth={2} strokeDasharray="4 3" />
        <line x1={ORIGIN} y1={ORIGIN} x2={wxs[0]} y2={wxs[1]} stroke="#5b53e0" strokeWidth={3} markerEnd="url(#xm)" />
        <text x={ihat[0] + 6} y={ihat[1] - 4} fontSize="13" fill="#2e9e6b" fontWeight="700">î</text>
        <text x={jhat[0] + 6} y={jhat[1] - 4} fontSize="13" fill="#e0533d" fontWeight="700">ĵ</text>
        <text x={wxs[0] + 6} y={wxs[1] - 4} fontSize="13" fill="#5b53e0" fontWeight="700">Wx</text>
      </svg>
      <Readout items={[
        ["W =", `[[${w00},${w01}],[${w10},${w11}]]`],
        ["x =", `(${x.x}, ${x.y})`],
        ["Wx =", `(${Wx[0].toFixed(2)}, ${Wx[1].toFixed(2)})`],
        ["det =", <span style={{ color: det < 0 ? "var(--bad)" : "var(--ink)" }}>{det.toFixed(2)}</span>],
      ]} />
      <div className="viz-controls">
        <div className="chips">
          {Object.keys(PRESETS).map((k) => (
            <button key={k} className="chip" onClick={() => setW(PRESETS[k])}>{k}</button>
          ))}
        </div>
        <Slider label="W[0,0]" value={w00} min={-3} max={3} step={0.1} onChange={(v) => setW([v, w01, w10, w11])} />
        <Slider label="W[0,1]" value={w01} min={-3} max={3} step={0.1} onChange={(v) => setW([w00, v, w10, w11])} />
        <Slider label="W[1,0]" value={w10} min={-3} max={3} step={0.1} onChange={(v) => setW([w00, w01, v, w11])} />
        <Slider label="W[1,1]" value={w11} min={-3} max={3} step={0.1} onChange={(v) => setW([w00, w01, w10, v])} />
        <Slider label="x" value={x.x} min={-3} max={3} step={0.5} onChange={(v) => setX({ ...x, x: v })} fmt={(v)=>`x=${v}`} />
        <Slider label="y" value={x.y} min={-3} max={3} step={0.5} onChange={(v) => setX({ ...x, y: v })} fmt={(v)=>`y=${v}`} />
      </div>
    </VizCard>
  );
}
