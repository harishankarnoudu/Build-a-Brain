import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createHashRouter, RouterProvider } from "react-router-dom";
import "katex/dist/katex.min.css";
import "./index.css";
import Layout from "./App.tsx";
import Home from "./pages/Home.tsx";
import LessonPage from "./pages/LessonPage.tsx";

// HashRouter so it works on any static host (GitHub Pages, internal server) with no config.
const router = createHashRouter([
  {
    path: "/",
    element: <Layout />,
    children: [
      { index: true, element: <Home /> },
      { path: "lesson/:id", element: <LessonPage /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>
);
