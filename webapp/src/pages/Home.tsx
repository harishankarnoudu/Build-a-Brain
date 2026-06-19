import { lazy, Suspense } from "react";
import { Link } from "react-router-dom";
import { PARTS } from "../lessons";

const NeuralNet3D = lazy(() => import("../three/NeuralNet3D"));

export default function Home() {
  let chapter = 0;
  return (
    <>
      <div className="hero">
        <div className="kicker">An interactive story</div>
        <h1>Build a Brain</h1>
        <p className="lead">One story, told end to end: we <strong>raise a mind</strong> — from a single bare
          number into a thinking machine. It learns to sense, to decide, to see, to read, to remember, and
          finally to <em>think</em>. And you don't just read it — you <strong>drag, orbit, and train</strong>
          every idea yourself. Roll a ball down a 3-D loss valley, grow neurons until they crack a puzzle, walk
          words through space. Learn by doing.</p>
        <div className="hero-cta">
          <Link to="/lesson/p00" className="btn">▶ Begin: the mind is born</Link>
          <Link to="/lesson/playground" className="btn ghost">🧪 Open the lab</Link>
        </div>
        <div className="hero-3d">
          <Suspense fallback={<div style={{ height: 320 }} />}>
            <NeuralNet3D />
          </Suspense>
        </div>
      </div>

      {PARTS.map((part) => (
        <section className="journey-part" key={part.name}>
          <div className="journey-head">
            <span className="journey-icon">{part.icon}</span>
            <div>
              <div className="journey-name">{part.name}</div>
              <div className="journey-blurb">{part.blurb}</div>
            </div>
          </div>
          <div className="grid-cards">
            {part.lessons.map((l) => {
              chapter++;
              return (
                <Link key={l.id} to={`/lesson/${l.id}`} className="card">
                  <div className="num">Chapter {String(chapter).padStart(2, "0")}{l.built ? " · ▶ live" : ""}</div>
                  <div className="title">{l.title}</div>
                  <div className="story">{l.story}</div>
                </Link>
              );
            })}
          </div>
        </section>
      ))}
    </>
  );
}
