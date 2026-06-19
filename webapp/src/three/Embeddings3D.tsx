import { useState, lazy, Suspense } from "react";
import { Html } from "@react-three/drei";
import { Arrow3D } from "./helpers";

const ThreeStage = lazy(() => import("./ThreeStage"));

type WP = { w: string; p: [number, number, number]; theme: "animal" | "royal" };
const WORDS: WP[] = [
  { w: "cat", p: [-2.0, -1.0, -0.5], theme: "animal" },
  { w: "dog", p: [-1.6, -1.4, -0.9], theme: "animal" },
  { w: "kitten", p: [-2.3, -0.6, -0.2], theme: "animal" },
  { w: "pet", p: [-1.4, -1.0, -1.2], theme: "animal" },
  { w: "man", p: [0.8, 0.8, 0.4], theme: "royal" },
  { w: "woman", p: [0.8, 2.0, 0.4], theme: "royal" },
  { w: "king", p: [2.0, 0.8, 1.0], theme: "royal" },
  { w: "queen", p: [2.0, 2.0, 1.0], theme: "royal" },
];

function Word({ w, p, theme }: WP) {
  return (
    <group position={p}>
      <mesh><sphereGeometry args={[0.11, 20, 20]} />
        <meshStandardMaterial color={theme === "animal" ? "#4f9dff" : "#ff7a59"} emissive={theme === "animal" ? "#4f9dff" : "#ff7a59"} emissiveIntensity={0.35} /></mesh>
      <Html distanceFactor={9} style={{ pointerEvents: "none", color: "#fff", fontWeight: 700, fontSize: 14, textShadow: "0 1px 4px #000", whiteSpace: "nowrap" }}>{w}</Html>
    </group>
  );
}

export default function Embeddings3D() {
  const [analogy, setAnalogy] = useState(true);
  const king = WORDS.find((x) => x.w === "king")!.p, man = WORDS.find((x) => x.w === "man")!.p, woman = WORDS.find((x) => x.w === "woman")!.p;
  const res: [number, number, number] = [king[0] - man[0] + woman[0], king[1] - man[1] + woman[1], king[2] - man[2] + woman[2]];

  return (
    <div className="viz dark">
      <Suspense fallback={<div style={{ height: 380, display: "grid", placeItems: "center", color: "#9d94ff" }}>loading 3D…</div>}>
        <ThreeStage camera={[4.5, 3, 4.5]} height={420}>
          {WORDS.map((x) => <Word key={x.w} {...x} />)}
          {analogy && <>
            <Arrow3D from={man} to={woman} color="#22d3c5" />
            <Arrow3D from={king} to={res} color="#22d3c5" />
          </>}
        </ThreeStage>
      </Suspense>
      <div className="viz-controls">
        <div className="ctl-row">
          <button className={"btn" + (analogy ? "" : " ghost")} onClick={() => setAnalogy((v) => !v)}>
            {analogy ? "Hide" : "Show"} analogy: king − man + woman ≈ queen</button>
          <span style={{ color: "#c8c8e0", fontSize: ".85rem" }}>drag to orbit</span>
        </div>
        <span style={{ color: "#c8c8e0", fontSize: ".9rem" }}>
          Orbit the cloud: <b style={{ color: "#4f9dff" }}>animals</b> cluster in one region,
          <b style={{ color: "#ff7a59" }}> royalty</b> in another — meaning becomes <em>position</em>. The two
          teal arrows are parallel, so the "man→woman" direction added to <b>king</b> lands right on
          <b> queen</b>. Relationships are directions in this space.</span>
      </div>
    </div>
  );
}
