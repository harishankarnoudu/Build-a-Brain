import type { ReactNode } from "react";

// A real-world "scene" opener — every chapter starts here, on something the reader
// already knows, before any math. `icon` is an emoji; `children` is the vivid moment.
export function Scene({ icon = "🎬", children }: { icon?: string; children: ReactNode }) {
  return (
    <div className="scene">
      <span className="scene-icon" aria-hidden>{icon}</span>
      <div className="scene-body">{children}</div>
    </div>
  );
}

// A "Your mission" quest box that turns a figure into a playground with goals.
export function Mission({ title = "Your mission", children }: { title?: string; children: ReactNode }) {
  return (
    <div className="mission">
      <div className="mission-head">🎯 {title}</div>
      <ul className="mission-list">{children}</ul>
    </div>
  );
}

// A small inline "in the real world" aside — connects an idea back to life.
export function RealWorld({ children }: { children: ReactNode }) {
  return (
    <div className="realworld"><span className="realworld-tag">In the real world</span> {children}</div>
  );
}
