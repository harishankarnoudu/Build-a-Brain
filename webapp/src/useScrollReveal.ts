import { useEffect } from "react";

// Scrollytelling: fade/slide article blocks in as they enter the viewport.
// Re-runs whenever `key` changes (i.e. on lesson navigation).
export function useScrollReveal(key: string) {
  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const blocks = document.querySelectorAll<HTMLElement>(
      ".content-inner h2, .content-inner .viz, .content-inner .callout, .content-inner .hand, .content-inner .mathblock, .content-inner ul"
    );
    blocks.forEach((el) => el.classList.add("reveal"));
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }),
      { threshold: 0.12, rootMargin: "0px 0px -8% 0px" }
    );
    blocks.forEach((el) => io.observe(el));
    // anything already on-screen at load reveals immediately
    requestAnimationFrame(() => blocks.forEach((el) => {
      const r = el.getBoundingClientRect();
      if (r.top < window.innerHeight) el.classList.add("in");
    }));
    return () => io.disconnect();
  }, [key]);
}
