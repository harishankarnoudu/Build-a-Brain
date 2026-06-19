# TinyML Explain — interactive web app

An interactive, browser-based version of the playbook, inspired by
[MLU-Explain](https://mlu-explain.github.io/), the
[TensorFlow Playground](https://playground.tensorflow.org/), and
[3Blue1Brown](https://www.3blue1brown.com/). React + Vite + TypeScript, fully client-side
(the neural network trains in your browser — no backend).

## Run it

```bash
cd webapp
npm install        # already done once
npm run dev        # http://localhost:5173/
```

Build a static bundle (deployable to GitHub Pages or any internal static host):

```bash
npm run build      # outputs to dist/
npm run preview    # preview the production build
```

Uses a hash router (`/#/lesson/...`) so it works on any static host with no server config.

## Content

**All 31 lessons have full article content** — no placeholders. **14 are fully interactive**, the
rest are complete prose articles (with typeset KaTeX math) in `src/pages/articles.tsx`.

| Route | Lesson | Interaction |
|-------|--------|-------------|
| `p01-dot-product` | The Dot Product | drag two vectors; value, angle, projection |
| `p01-matrix` | A Matrix Transforms Space | sliders/presets deform the grid & unit square |
| `p02-gradient-descent` | Gradient Descent | animate the ball down the loss bowl; tune lr |
| `p03` | Probability & Cross-Entropy | drag a distribution; watch entropy |
| `p05` | The ML Mindset | slide polynomial degree; watch overfitting |
| `p07` | Logistic Regression | train a live boundary; fails on non-linear data |
| `p10` | Neural Networks | add hidden neurons until XOR is cracked |
| `p11` | Backpropagation | step a computation graph forward & backward |
| `p12` | Optimizers | race SGD/Momentum/RMSProp/Adam on a loss surface |
| `playground` | Neural Network Playground | train an MLP in-browser; live decision boundary |
| `p15` | Convolutional Networks | slide a kernel; watch the feature map form |
| `p18` | Word Embeddings | click words for cosine; king−man+woman analogy |
| `p24` | RLHF / DPO | tune the preference loss |
| `p26` | Prompting & Decoding | temperature / top-k / top-p, then sample |

(Access any lesson at `/#/lesson/<route>`.) Prose articles cover P00, P04, P06, P08, P09, P13,
P16, P17, P19, P20, P21, P22, P23, P25, P27, P28, P29.

## Deploy

The build is host-agnostic (`base: "./"` + hash router):

```bash
npm run build           # -> dist/  (deploy to any static host)
```

- **GitHub Pages:** push `dist/` to a `gh-pages` branch (or commit `dist/` and point Pages at it).
- **Internal static server / nginx:** serve the `dist/` folder.
- **Quick local check:** `npm run preview`.

## Structure

```
src/
  main.tsx              router (hash router)
  App.tsx               layout shell + sidebar nav
  index.css             design system
  lessons.ts            curriculum metadata (all 30)
  components/ui.tsx     Slider, VizCard, Readout, Chips
  viz/nn.ts             from-scratch MLP (forward + backprop) + toy datasets
  lessons/              the 4 interactive components
  pages/                Home, LessonPage, lessonContent (prose + interactives)
```

## Adding the next lesson

1. Build an interactive in `src/lessons/`.
2. Add its prose + the component to `CONTENT` in `src/pages/lessonContent.tsx`.
3. Set `built: true` on its entry in `src/lessons.ts`.
