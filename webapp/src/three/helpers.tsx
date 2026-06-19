import { useMemo } from "react";
import { Line } from "@react-three/drei";
import * as THREE from "three";

type V3 = [number, number, number];

// A 3-D arrow: a line shaft from `from` to `to` plus a cone arrowhead aligned to the direction.
export function Arrow3D({ from = [0, 0, 0], to, color, width = 2.5 }: { from?: V3; to: V3; color: string; width?: number }) {
  const { tip, quat, len } = useMemo(() => {
    const a = new THREE.Vector3(...from), b = new THREE.Vector3(...to);
    const dir = b.clone().sub(a); const length = dir.length();
    const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
    return { tip: b, quat: q, len: length };
  }, [from, to]);
  return (
    <>
      <Line points={[from, to]} color={color} lineWidth={width} />
      <mesh position={tip} quaternion={quat}>
        <coneGeometry args={[0.06, 0.18, 16]} />
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.3} />
      </mesh>
      {len < 0.0001 && null}
    </>
  );
}
