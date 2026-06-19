import type { ReactNode } from "react";

export function Slider({
  label, value, min, max, step = 0.1, onChange, fmt = (v) => v.toFixed(2),
}: {
  label: string; value: number; min: number; max: number; step?: number;
  onChange: (v: number) => void; fmt?: (v: number) => string;
}) {
  return (
    <div className="ctl">
      <label>{label}</label>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))} />
      <output>{fmt(value)}</output>
    </div>
  );
}

export function VizCard({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={"viz " + className}>{children}</div>;
}

export function Readout({ items }: { items: [string, ReactNode][] }) {
  return (
    <div className="viz-readout">
      {items.map(([k, v], i) => (
        <span key={i}>{k} <b>{v}</b></span>
      ))}
    </div>
  );
}

export function Chips<T extends string>({
  options, value, onChange,
}: { options: { id: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="chips">
      {options.map((o) => (
        <button key={o.id} className={"chip" + (o.id === value ? " active" : "")}
          onClick={() => onChange(o.id)}>{o.label}</button>
      ))}
    </div>
  );
}
