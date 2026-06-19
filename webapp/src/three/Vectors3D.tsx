import { useState, lazy, Suspense } from "react";
import { Grid } from "@react-three/drei";
import { Arrow3D } from "./helpers";
import { Slider, Readout } from "../components/ui";

const ThreeStage = lazy(() => import("./ThreeStage"));

// Two vectors in 3-D space; see the dot product and angle update as you move b.
export default function Vectors3D() {
  const [a] = useState<[number, number, number]>([1.6, 1.2, 0.6]);
  const [b, setB] = useState<[number, number, number]>([1.4, 0.2, 1.6]);
  const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const na = Math.hypot(...a), nb = Math.hypot(...b);
  const angle = (Math.acos(Math.max(-1, Math.min(1, dot / (na * nb || 1)))) * 180) / Math.PI;
  const dotColor = dot > 0.01 ? "var(--good)" : dot < -0.01 ? "var(--bad)" : "var(--ink-faint)";

  return (
    <div className="viz dark">
      <Suspense fallback={<div style={{ height: 380, display: "grid", placeItems: "center", color: "#9d94ff" }}>loading 3D…</div>}>
        <ThreeStage camera={[3, 2.4, 3]}>
          <Grid args={[6, 6]} cellColor="#2a2c4a" sectionColor="#3a3c6a" infiniteGrid fadeDistance={14} position={[0, 0, 0]} />
          <Arrow3D to={a} color="#ff5d3b" />
          <Arrow3D to={b} color="#4f9dff" />
        </ThreeStage>
      </Suspense>
      <Readout items={[
        ["a", `(${a.map((v) => v.toFixed(1)).join(", ")})`],
        ["b", `(${b.map((v) => v.toFixed(1)).join(", ")})`],
        ["a · b", <span style={{ color: dotColor }}>{dot.toFixed(2)}</span>],
        ["angle", `${angle.toFixed(0)}°`],
      ]} />
      <div className="viz-controls">
        <Slider label="b.x" value={b[0]} min={-2} max={2} step={0.1} onChange={(v) => setB([v, b[1], b[2]])} />
        <Slider label="b.y" value={b[1]} min={-2} max={2} step={0.1} onChange={(v) => setB([b[0], v, b[2]])} />
        <Slider label="b.z" value={b[2]} min={-2} max={2} step={0.1} onChange={(v) => setB([b[0], b[1], v])} />
        <span style={{ color: "#c8c8e0", fontSize: ".9rem" }}>
          Orbit the scene and move <b style={{ color: "#4f9dff" }}>b</b>. The dot product is biggest when the
          arrows align, zero when perpendicular (90°), negative when they oppose — the same rule as 2-D, now
          in real space. This is how attention scores "how related" two token-vectors are.</span>
      </div>
    </div>
  );
}
