import { useEffect, useMemo, useRef, useState, lazy, Suspense } from "react";
import { useFrame } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import { Slider } from "../components/ui";
import { Transport } from "../components/Controls";
import { usePersistentState } from "../store";

const ThreeStage = lazy(() => import("./ThreeStage"));

// True loss drives the ball; a gentle vertical scale keeps the whole valley in frame.
const RANGE = 1.9, SEG = 44, VSCALE = 0.34;
const loss = (x: number, z: number) => x * x + 1.4 * z * z;       // gradient = [2x, 2.8z]
const grad = (x: number, z: number): [number, number] => [2 * x, 2.8 * z];
const hy = (x: number, z: number) => VSCALE * loss(x, z);          // display height

function Surface() {
  const geo = useMemo(() => {
    const g = new THREE.PlaneGeometry(2 * RANGE, 2 * RANGE, SEG, SEG);
    const pos = g.attributes.position as THREE.BufferAttribute;
    const colors: number[] = [];
    const lo = new THREE.Color("#2bd4c4"), hi = new THREE.Color("#3a2f8f");
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i), y = pos.getY(i);
      const h = hy(x, y); pos.setZ(i, h);
      const t = Math.min(1, h / (VSCALE * 8));
      const c = lo.clone().lerp(hi, t); colors.push(c.r, c.g, c.b);
    }
    g.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    g.computeVertexNormals(); g.rotateX(-Math.PI / 2);
    return g;
  }, []);
  return (
    <>
      <mesh geometry={geo}><meshStandardMaterial vertexColors transparent opacity={0.95} side={THREE.DoubleSide} /></mesh>
      <mesh geometry={geo}><meshBasicMaterial wireframe color="#ffffff" transparent opacity={0.06} /></mesh>
    </>
  );
}

function Ball({ lr, playing, stepSig }: { lr: number; playing: boolean; stepSig: number }) {
  const ref = useRef<THREE.Mesh>(null);
  const p = useRef<[number, number]>([-1.7, 1.4]);
  const [trail, setTrail] = useState<[number, number, number][]>([[-1.7, hy(-1.7, 1.4) + 0.06, 1.4]]);
  const acc = useRef(0);

  const doStep = () => {
    const [x, z] = p.current; const [gx, gz] = grad(x, z);
    p.current = [x - lr * gx, z - lr * gz];
    setTrail((t) => [...t.slice(-140), [p.current[0], hy(...p.current) + 0.06, p.current[1]]]);
  };
  // manual single-step when the parent bumps stepSig (effect, never during render)
  useEffect(() => { if (stepSig >= 0) doStep(); }, [stepSig]); // eslint-disable-line

  useFrame((_, dt) => {
    if (playing) { acc.current += dt; if (acc.current > 0.06) { acc.current = 0; doStep(); } }
    if (ref.current) { const [x, z] = p.current; ref.current.position.set(x, hy(x, z) + 0.1, z); }
  });

  return (
    <>
      <mesh ref={ref}><sphereGeometry args={[0.1, 24, 24]} /><meshStandardMaterial color="#ff5d3b" emissive="#ff5d3b" emissiveIntensity={0.5} /></mesh>
      {trail.length > 1 && <Line points={trail} color="#ffb27a" lineWidth={2.5} />}
    </>
  );
}

export default function GradientDescent3D() {
  const [lr, setLr] = usePersistentState("gd3d:lr", 0.12);
  const [playing, setPlaying] = usePersistentState("gd3d:playing", false);
  const [resetKey, setResetKey] = useState(0);
  const [stepSig, setStepSig] = useState(-1);

  return (
    <div className="viz dark">
      <Suspense fallback={<div style={{ height: 400, display: "grid", placeItems: "center", color: "#9d94ff" }}>loading 3D…</div>}>
        <ThreeStage camera={[4.4, 3.4, 4.4]} target={[0, 0.7, 0]} height={400} maxDistance={13}>
          <Surface />
          <Ball key={resetKey} lr={lr} playing={playing} stepSig={stepSig} />
          <mesh position={[0, 0.08, 0]}><sphereGeometry args={[0.07, 16, 16]} /><meshStandardMaterial color="#22d3c5" emissive="#22d3c5" emissiveIntensity={0.7} /></mesh>
        </ThreeStage>
      </Suspense>
      <div className="viz-controls">
        <Transport playing={playing} onPlay={() => setPlaying((p) => !p)}
          onStep={() => setStepSig((s) => s + 1)}
          onReset={() => { setPlaying(false); setResetKey((k) => k + 1); setStepSig(-1); }}
          status={<span style={{ color: "#c8c8e0" }}>drag to orbit · scroll to zoom</span>} />
        <Slider label="learning rate" value={lr} min={0.02} max={1.1} step={0.02} onChange={setLr} />
        <span style={{ color: "#c8c8e0", fontSize: ".9rem" }}>
          The orange ball follows the slope downhill to the teal minimum — <b>gradient descent</b> on a real
          loss valley. Press <b>Step</b> to move one update at a time, or <b>Play</b> to roll. Small rates
          crawl; large rates overshoot the walls and bounce.</span>
      </div>
    </div>
  );
}
