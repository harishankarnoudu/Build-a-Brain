import { useState, type ReactNode } from "react";

// Collapsible "Show the math" block — keeps visuals front-and-centre, math one click away.
export function ShowMath({ children, label = "Show the math" }: { children: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className={"showmath" + (open ? " open" : "")}>
      <button className="showmath-toggle" onClick={() => setOpen((o) => !o)}>
        <span>{open ? "▾" : "▸"}</span> {open ? "Hide the math" : label}
      </button>
      {open && <div className="showmath-body">{children}</div>}
    </div>
  );
}
