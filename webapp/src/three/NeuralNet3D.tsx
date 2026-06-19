import { useMemo, useRef, useState, lazy, Suspense } from "react";
import { useFrame } from "@react-three/fiber";
import { Line } from "@react-three/drei";
import * as THREE from "three";
import { Transport } from "../components/Controls";
import { usePersistentState } from "../store";

const ThreeStage = lazy(() => import("./ThreeStage"));
const LAYERS = [3, 5, 5, 2];

type Neuron = { pos: [number, number, number]; layer: number };
function build() {
  const neurons: Neuron[] = [];
  const edges: [[number, number, number], [number, number, number]][] = [];
  const xGap = 1.5, x0 = -((LAYERS.length - 1) * xGap) / 2;
  const layerPts: [number, number, number][][] = LAYERS.map((n, li) =>
    Array.from({ length: n }, (_, i) => {
      const p: [number, number, number] = [x0 + li * xGap, (i - (n - 1) / 2) * 0.75, 0];
      neurons.push({ pos: p, layer: li });
      return p;
    })
  );
  for (let l = 0; l < LAYERS.length - 1; l++)
    for (const a of layerPts[l]) for (const b of layerPts[l + 1]) edges.push([a, b]);
  return { neurons, edges };
}

function Net({ playing, stepSig }: { playing: boolean; stepSig: number }) {
  const { neurons, edges } = useMemo(build, []);
  const refs = useRef<(THREE.Mesh | null)[]>([]);
  const front = useRef(0);
  const lastStep = useRef(stepSig);

  useFrame((_, dt) => {
    if (stepSig !== lastStep.current) { lastStep.current = stepSig; front.current = (front.current + 1) % (LAYERS.length + 1.2); }
    else if (playing) front.current = (front.current + dt * 0.9) % (LAYERS.length + 1.2);
    neurons.forEach((n, i) => {
      const m = refs.current[i]; if (!m) return;
      const d = Math.abs(n.layer - front.current);
      const glow = Math.max(0, 1 - d * 1.4);
      (m.material as THREE.MeshStandardMaterial).emissiveIntensity = 0.25 + glow * 1.6;
      const s = 1 + glow * 0.5; m.scale.setScalar(s);
    });
  });

  return (
    <>
      {edges.map((e, i) => <Line key={i} points={e} color="#3a3c6a" lineWidth={1} transparent opacity={0.4} />)}
      {neurons.map((n, i) => (
        <mesh key={i} position={n.pos} ref={(el) => { refs.current[i] = el; }}>
          <sphereGeometry args={[0.18, 24, 24]} />
          <meshStandardMaterial color="#9d94ff" emissive="#9d94ff" emissiveIntensity={0.3} />
        </mesh>
      ))}
    </>
  );
}

export default function NeuralNet3D() {
  const [playing, setPlaying] = usePersistentState("nn3d:playing", false);
  const [stepSig, setStepSig] = useState(0);
  return (
    <div className="viz dark">
      <Suspense fallback={<div style={{ height: 380, display: "grid", placeItems: "center", color: "#9d94ff" }}>loading 3D…</div>}>
        <ThreeStage camera={[0.5, 1.5, 6]} target={[0, 0, 0]}>
          <Net playing={playing} stepSig={stepSig} />
        </ThreeStage>
      </Suspense>
      <div className="viz-controls">
        <Transport playing={playing} onPlay={() => setPlaying((p) => !p)}
          onStep={() => setStepSig((s) => s + 1)} onReset={() => setPlaying(false)}
          status={<span style={{ color: "#c8c8e0" }}>drag to orbit</span>} />
        <span style={{ color: "#c8c8e0", fontSize: ".9rem" }}>
          A network is <b>layers of neurons</b> wired together. Press <b>Play</b> and watch the activation wave
          flow left→right: each layer lights up as the signal (the <b>forward pass</b>) passes through it.
          <b> Step</b> advances one layer at a time. This is the shape of the net you train in the lab.</span>
      </div>
    </div>
  );
}
