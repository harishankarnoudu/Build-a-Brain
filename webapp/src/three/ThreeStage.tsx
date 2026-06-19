import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import type { ReactNode } from "react";

// Shared 3D stage: a configured canvas with orbit controls + lighting.
// Lazy-loaded at the lesson level so Three.js never bloats non-3D pages.
export default function ThreeStage({
  children, camera = [3.2, 2.6, 3.2], target = [0, 0, 0], height = 380, dark = true, maxDistance = 11,
}: { children: ReactNode; camera?: [number, number, number]; target?: [number, number, number]; height?: number; dark?: boolean; maxDistance?: number }) {
  return (
    <div style={{ height, borderRadius: 12, overflow: "hidden", background: dark ? "#0f1020" : "#fbfaf6" }}>
      <Canvas camera={{ position: camera, fov: 50 }} dpr={[1, 2]}>
        <color attach="background" args={[dark ? "#0f1020" : "#fbfaf6"]} />
        <ambientLight intensity={0.75} />
        <directionalLight position={[5, 8, 5]} intensity={1.1} />
        <directionalLight position={[-5, 3, -4]} intensity={0.45} color="#9d94ff" />
        {children}
        <OrbitControls enablePan={false} target={target} minDistance={2.4} maxDistance={maxDistance} />
      </Canvas>
    </div>
  );
}
