import katex from "katex";

// NOTE: the `children` passed here are ALWAYS author-written static LaTeX literals from the
// lesson source — never user input. KaTeX renders to safe markup (no script execution), so
// dangerouslySetInnerHTML carries no XSS risk for this trusted, compile-time content.

// Inline math:  <M>{"a \\cdot b"}</M>
export function M({ children }: { children: string }) {
  const html = katex.renderToString(children, { throwOnError: false, displayMode: false });
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

// Display (block) math, centered on its own line.
export function MathBlock({ children }: { children: string }) {
  const html = katex.renderToString(children, { throwOnError: false, displayMode: true });
  return <div className="mathblock" dangerouslySetInnerHTML={{ __html: html }} />;
}
