import type { ReactNode } from "react";
import { usePersistentState } from "../store";

// A consistent transport bar: Play/Pause + optional Step + Reset + a live status line.
export function Transport({
  playing, onPlay, onStep, onReset, status,
}: {
  playing: boolean; onPlay: () => void; onStep?: () => void; onReset: () => void; status?: ReactNode;
}) {
  return (
    <div className="transport">
      <button className="btn" onClick={onPlay}>{playing ? "⏸ Pause" : "▶ Play"}</button>
      {onStep && <button className="btn ghost" onClick={onStep} disabled={playing}>⏭ Step</button>}
      <button className="btn ghost" onClick={onReset}>↺ Reset</button>
      {status != null && <span className="transport-status">{status}</span>}
    </div>
  );
}

// 2D / 3D (and "more") view tabs. Selection persists per figure via the store.
export function ViewSwitcher({
  id, views,
}: { id: string; views: { key: string; label: string; node: ReactNode }[] }) {
  const [active, setActive] = usePersistentState(`view:${id}`, views[0].key);
  const current = views.find((v) => v.key === active) ?? views[0];
  return (
    <div>
      <div className="view-tabs">
        {views.map((v) => (
          <button key={v.key} className={"view-tab" + (v.key === current.key ? " active" : "")}
            onClick={() => setActive(v.key)}>{v.label}</button>
        ))}
      </div>
      {current.node}
    </div>
  );
}
