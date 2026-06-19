import { useState, useEffect } from "react";
import { Link, NavLink, Outlet, useLocation } from "react-router-dom";
import { PARTS, ALL_LESSONS } from "./lessons";

export default function Layout() {
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); window.scrollTo(0, 0); }, [loc.pathname]);

  let chapter = 0;
  return (
    <div className="shell">
      <button className="menu-btn" aria-label="menu" onClick={() => setOpen((o) => !o)}>☰</button>
      <div className={"scrim" + (open ? " show" : "")} onClick={() => setOpen(false)} />
      <aside className={"sidebar" + (open ? " open" : "")}>
        <Link to="/" className="brand">🧠 Build&nbsp;a&nbsp;Brain</Link>
        <div className="tagline">From one number to a thinking machine</div>
        {PARTS.map((part) => (
          <div className="nav-part" key={part.name}>
            <div className="nav-part-title"><span className="nav-part-icon">{part.icon}</span> {part.name}</div>
            {part.lessons.map((l) => {
              chapter++;
              const ch = chapter;
              return (
                <NavLink key={l.id} to={`/lesson/${l.id}`}
                  className={({ isActive }) => "nav-link" + (l.built ? "" : " soon") + (isActive ? " active" : "")}>
                  <span className="nav-num">{String(ch).padStart(2, "0")}</span>
                  <span className="nav-title">{l.title}</span>
                  {l.built && <span className="nav-badge">▶</span>}
                </NavLink>
              );
            })}
          </div>
        ))}
        <div className="sidebar-foot">{ALL_LESSONS.length} chapters · {ALL_LESSONS.filter((l) => l.built).length} interactive</div>
      </aside>
      <main className="content">
        <div className="content-inner"><Outlet /></div>
      </main>
    </div>
  );
}
