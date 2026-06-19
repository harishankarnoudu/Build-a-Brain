import { useRef, useState } from "react";
import { VizCard, Readout } from "../components/ui";
import { usePersistentState } from "../store";

// SVG playground: drag the tips of vectors a and b; see a·b, the angle, and b's projection onto a.
const SIZE = 420, ORIGIN = SIZE / 2, SCALE = 70; // 1 unit = 70px

function toScreen(x: number, y: number) { return [ORIGIN + x * SCALE, ORIGIN - y * SCALE]; }
function toMath(px: number, py: number) { return [(px - ORIGIN) / SCALE, (ORIGIN - py) / SCALE]; }

export default function DotProduct() {
  const [a, setA] = usePersistentState("dot2d:a", { x: 2, y: 1 });
  const [b, setB] = usePersistentState("dot2d:b", { x: 1, y: 2 });
  const [drag, setDrag] = useState<null | "a" | "b">(null);
  const svgRef = useRef<SVGSVGElement>(null);

  const dot = a.x * b.x + a.y * b.y;
  const na = Math.hypot(a.x, a.y), nb = Math.hypot(b.x, b.y);
  const cos = dot / (na * nb || 1);
  const angle = (Math.acos(Math.max(-1, Math.min(1, cos))) * 180) / Math.PI;
  // projection of b onto a
  const k = dot / (na * na || 1);
  const proj = { x: k * a.x, y: k * a.y };

  function pointerMove(e: React.PointerEvent) {
    if (!drag || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const [mx, my] = toMath(
      ((e.clientX - rect.left) / rect.width) * SIZE,
      ((e.clientY - rect.top) / rect.height) * SIZE
    );
    const v = { x: Math.round(mx * 2) / 2, y: Math.round(my * 2) / 2 };
    drag === "a" ? setA(v) : setB(v);
  }

  const [ax, ay] = toScreen(a.x, a.y);
  const [bx, by] = toScreen(b.x, b.y);
  const [px, py] = toScreen(proj.x, proj.y);
  const dotColor = dot > 0.01 ? "var(--good)" : dot < -0.01 ? "var(--bad)" : "var(--ink-faint)";

  return (
    <VizCard>
      <svg ref={svgRef} className="viz-stage" viewBox={`0 0 ${SIZE} ${SIZE}`} style={{ height: 420 }}
        onPointerMove={pointerMove} onPointerUp={() => setDrag(null)} onPointerLeave={() => setDrag(null)}>
        <defs>
          <marker id="arrowRed" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
            <path d="M0,0 L7,3 L0,6 Z" fill="#e0533d" /></marker>
          <marker id="arrowBlue" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto">
            <path d="M0,0 L7,3 L0,6 Z" fill="#2563eb" /></marker>
        </defs>
        {/* grid */}
        {Array.from({ length: 9 }, (_, i) => i - 4).map((g) => (
          <g key={g} stroke="#e6e1d6">
            <line x1={ORIGIN + g * SCALE} y1={0} x2={ORIGIN + g * SCALE} y2={SIZE} />
            <line x1={0} y1={ORIGIN + g * SCALE} x2={SIZE} y2={ORIGIN + g * SCALE} />
          </g>
        ))}
        <line x1={0} y1={ORIGIN} x2={SIZE} y2={ORIGIN} stroke="#b9b3a5" />
        <line x1={ORIGIN} y1={0} x2={ORIGIN} y2={SIZE} stroke="#b9b3a5" />
        {/* projection */}
        <line x1={bx} y1={by} x2={px} y2={py} stroke="#2e9e6b" strokeDasharray="4 4" />
        <line x1={ORIGIN} y1={ORIGIN} x2={px} y2={py} stroke="#2e9e6b" strokeWidth={5} opacity={0.45} />
        {/* vectors */}
        <line x1={ORIGIN} y1={ORIGIN} x2={ax} y2={ay} stroke="#e0533d" strokeWidth={3} markerEnd="url(#arrowRed)" />
        <line x1={ORIGIN} y1={ORIGIN} x2={bx} y2={by} stroke="#2563eb" strokeWidth={3} markerEnd="url(#arrowBlue)" />
        {/* draggable handles */}
        <circle cx={ax} cy={ay} r={11} fill="#e0533d" opacity={0.9} cursor="grab"
          onPointerDown={() => setDrag("a")} />
        <circle cx={bx} cy={by} r={11} fill="#2563eb" opacity={0.9} cursor="grab"
          onPointerDown={() => setDrag("b")} />
        <text x={ax + 10} y={ay - 8} fontSize="14" fill="#e0533d" fontWeight="700">a</text>
        <text x={bx + 10} y={by - 8} fontSize="14" fill="#2563eb" fontWeight="700">b</text>
        <text x={ORIGIN + 6} y={SIZE - 8} fontSize="20" fontWeight="700" fill={dotColor as string}>
          a · b = {dot.toFixed(2)}
        </text>
      </svg>
      <Readout items={[
        ["a =", `(${a.x}, ${a.y})`],
        ["b =", `(${b.x}, ${b.y})`],
        ["a · b =", <span style={{ color: dotColor }}>{dot.toFixed(2)}</span>],
        ["angle =", `${angle.toFixed(0)}°`],
        ["cosine =", cos.toFixed(2)],
      ]} />
      <div className="viz-controls" style={{ color: "var(--ink-soft)", fontSize: ".9rem" }}>
        <span>Drag the <b style={{ color: "#e0533d" }}>red</b> and <b style={{ color: "#2563eb" }}>blue</b> dots
        to feel two ideas <em>agree</em>, <em>ignore</em>, or <em>oppose</em> each other. The big number bottom-left
        is the verdict — the same one a recommender computes when it scores how well a song matches your taste.</span>
      </div>
    </VizCard>
  );
}
