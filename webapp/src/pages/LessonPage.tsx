import { Link, useParams } from "react-router-dom";
import { ALL_LESSONS, getLesson, lessonIndex, partOf, TOTAL } from "../lessons";
import { CONTENT } from "./lessonContent";
import { useScrollReveal } from "../useScrollReveal";

export default function LessonPage() {
  const { id = "" } = useParams();
  useScrollReveal(id);
  const lesson = getLesson(id);
  if (!lesson) return <p>Chapter not found. <Link to="/">Back to the journey</Link>.</p>;

  const idx = lessonIndex(id);
  const chapter = idx + 1;
  const prev = idx > 0 ? ALL_LESSONS[idx - 1] : null;
  const next = idx < ALL_LESSONS.length - 1 ? ALL_LESSONS[idx + 1] : null;
  const content = CONTENT[id];
  const part = partOf(id);

  return (
    <article>
      <div className="chapter-strip">
        <span className="chapter-badge">{part?.icon} Chapter {chapter} of {TOTAL}</span>
        <div className="chapter-progress"><div style={{ width: `${(chapter / TOTAL) * 100}%` }} /></div>
      </div>
      {prev && <p className="prev-recap">Previously — <em>{prev.story}</em></p>}
      <div className="kicker">{lesson.code} · {part?.name.replace(/^Part \d+ · /, "")}</div>
      <h1>{lesson.title}</h1>
      <p className="story-lead">“{lesson.story}”</p>

      {content ?? (
        <>
          <p className="lead">{lesson.desc}</p>
          <div className="callout">
            <strong>This chapter's interactive is on the way.</strong> The full walkthrough — plain-words
            explanation, a worked example by hand, and verifying code — lives in the playbook notebook
            <code> {lesson.code}</code>. Meanwhile, jump into a live chapter below.
          </div>
        </>
      )}

      <div className="story-nav">
        {prev
          ? <Link to={`/lesson/${prev.id}`} className="story-nav-card prev"><span className="snc-label">← Previously</span><span className="snc-title">{prev.title}</span></Link>
          : <Link to="/" className="story-nav-card prev"><span className="snc-label">←</span><span className="snc-title">The journey</span></Link>}
        {next
          ? <Link to={`/lesson/${next.id}`} className="story-nav-card next"><span className="snc-label">Next in the story →</span><span className="snc-title">{next.title}</span><span className="snc-story">{next.story}</span></Link>
          : <Link to="/" className="story-nav-card next"><span className="snc-label">Journey complete →</span><span className="snc-title">You built a brain 🎉</span></Link>}
      </div>
    </article>
  );
}
