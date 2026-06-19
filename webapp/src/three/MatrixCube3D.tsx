import { lazy, Suspense } from "react";
import { Line } from "@react-three/drei";
import { Arrow3D } from "./helpers";
import { Slider, Readout } from "../components/ui";
import { usePersistentState } from "../store";

const ThreeStage = lazy(() => import("./ThreeStage"));
type V = [number, number, number];

// A 3x3 matrix transforms a unit cube + the basis vectors î ĵ k̂ — orbitable.
const CUBE: V[] = [[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
const EDGES: [number, number][] = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
const PRESETS: Record<string, number[]> = {
  identity: [1,0,0, 0,1,0, 0,0,1],
  scale:    [1.6,0,0, 0,1.6,0, 0,0,1.6],
  shear:    [1,0.8,0, 0,1,0, 0,0,1],
  rotateY:  [Math.cos(0.9),0,Math.sin(0.9), 0,1,0, -Math.sin(0.9),0,Math.cos(0.9)],
};

export default function MatrixCube3D() {
  const [W, setW] = usePersistentState<number[]>("mat3d:W", [1.2,0.5,0, 0,1.1,0.4, 0.3,0,1]);
  const apply = (v: V): V => [
    W[0]*v[0] + W[1]*v[1] + W[2]*v[2],
    W[3]*v[0] + W[4]*v[1] + W[5]*v[2],
    W[6]*v[0] + W[7]*v[1] + W[8]*v[2],
  ];
  const det =
    W[0]*(W[4]*W[8]-W[5]*W[7]) - W[1]*(W[3]*W[8]-W[5]*W[6]) + W[2]*(W[3]*W[7]-W[4]*W[6]);
  const tc = CUBE.map(apply);
  const center: V = [0.5,0.5,0.5];
  const cc = apply(center);

  return (
    <div className="viz dark">
      <Suspense fallback={<div style={{ height: 400, display: "grid", placeItems: "center", color: "#9d94ff" }}>loading 3D…</div>}>
        <ThreeStage camera={[3.4, 2.8, 3.8]} target={[cc[0], cc[1], cc[2]]} height={400}>
          {/* original cube (faint) */}
          {EDGES.map(([a, b], i) => <Line key={"o"+i} points={[CUBE[a], CUBE[b]]} color="#3a3c6a" lineWidth={1} />)}
          {/* transformed cube */}
          {EDGES.map(([a, b], i) => <Line key={"t"+i} points={[tc[a], tc[b]]} color="#9d94ff" lineWidth={2.5} />)}
          {/* basis vectors = columns of W */}
          <Arrow3D to={[W[0], W[3], W[6]]} color="#ff5d3b" />
          <Arrow3D to={[W[1], W[4], W[7]]} color="#2bd4c4" />
          <Arrow3D to={[W[2], W[5], W[8]]} color="#4f9dff" />
        </ThreeStage>
      </Suspense>
      <Readout items={[
        ["det", <span style={{ color: det < 0 ? "var(--bad)" : "var(--ink)" }}>{det.toFixed(2)}</span>],
        ["volume scale", `${Math.abs(det).toFixed(2)}×`],
      ]} />
      <div className="viz-controls">
        <div className="chips">
          {Object.keys(PRESETS).map((k) => <button key={k} className="chip" onClick={() => setW(PRESETS[k])}>{k}</button>)}
        </div>
        <Slider label="x-scale W[0,0]" value={W[0]} min={-2} max={2} step={0.1} onChange={(v) => setW(W.map((x, i) => i === 0 ? v : x))} />
        <Slider label="y-scale W[1,1]" value={W[4]} min={-2} max={2} step={0.1} onChange={(v) => setW(W.map((x, i) => i === 4 ? v : x))} />
        <Slider label="z-scale W[2,2]" value={W[8]} min={-2} max={2} step={0.1} onChange={(v) => setW(W.map((x, i) => i === 8 ? v : x))} />
        <Slider label="shear W[0,1]" value={W[1]} min={-1.5} max={1.5} step={0.1} onChange={(v) => setW(W.map((x, i) => i === 1 ? v : x))} />
        <span style={{ color: "#c8c8e0", fontSize: ".9rem" }}>
          The faint cube is the original; the violet cube is where the matrix sends it. The three arrows are
          the columns of <b>W</b> — where the axes î ĵ k̂ land. <b>det</b> is how much volume is scaled (negative
          = space flipped inside-out). Try the presets, then orbit.</span>
      </div>
    </div>
  );
}
