# 🧠 Build a Brain

> **An interactive, story-driven journey from a single number to a thinking machine.**

### 🔗 [**Open the live site → harishankarnoudu.github.io/Build-a-Brain**](https://harishankarnoudu.github.io/Build-a-Brain/)

[![Deploy to GitHub Pages](https://github.com/harishankarnoudu/Build-a-Brain/actions/workflows/deploy.yml/badge.svg)](https://github.com/harishankarnoudu/Build-a-Brain/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Most ML courses hand you formulas. **Build a Brain** tells one continuous story — you *raise a
mind*, and watch it learn to sense, decide, see, read, remember, and finally think. Every idea is
something you can **drag, orbit, and train**, and every formula is earned the same way: plain words →
a tiny worked example **by hand** → code (or a live figure) that verifies the numbers.

It comes in two halves that tell the same story at two depths:

1. **The Web App** (`webapp/`) — an interactive, 3Blue1Brown-style site you explore in the browser.
2. **The Notebooks** (`playbook/` + `notebooks/`) — the deep, runnable version, where you build a
   real GPT from scratch.

Inspired by [MLU-Explain](https://mlu-explain.github.io/),
[TensorFlow Playground](https://playground.tensorflow.org/), and
[3Blue1Brown](https://www.3blue1brown.com/).

---

## ✨ The story (the curriculum)

The whole journey is one arc — *raising a mind* — across seven life stages:

| Part | Stage | What the mind gains |
|------|-------|---------------------|
| **0 · Foundations** | *Born* | numbers, the dot product (its first sense), matrices, gradient descent, probability, data |
| **1 · Classical ML** | *First instincts* | regression, classification, the temptation to memorise, honest evaluation |
| **2 · Deep Learning** | *A brain forms* | neurons, backprop (the flash of learning), optimizers, training tricks, CNNs (eyes), RNNs (memory) |
| **3 · Language** | *Learning to mean* | classical NLP, word embeddings (meaning as direction), attention |
| **4 · Transformers** | *Its true form* | every piece assembled into the transformer |
| **5 · LLM Engineering** | *Raising a real mind* | pretraining, fine-tuning, LoRA, RLHF/DPO, RAG & agents, decoding, eval & safety |
| **6 · Deployment** | *Out into the world* | serving (KV-cache), MLOps |

See [`STORY_GUIDE.md`](STORY_GUIDE.md) for the full narrative spine and writing philosophy, and
[`PLAYBOOK_ROADMAP.md`](PLAYBOOK_ROADMAP.md) / [`TINYGPT_LEARNING_ROADMAP.md`](TINYGPT_LEARNING_ROADMAP.md)
for the detailed plans.

---

## 🚀 Quick start — the web app

No install needed to explore it — just open the **[live site](https://harishankarnoudu.github.io/Build-a-Brain/)**.
To run or hack on it locally: it's React + Vite + TypeScript + Three.js, fully client-side (the neural
network trains in your browser — no backend), on a single fixed port, **6023**.

```bash
cd webapp
npm install        # one-time
npm run dev        # → http://localhost:6023
```

Other commands:

```bash
npm run build      # type-check + bundle to webapp/dist (host-agnostic: relative base + hash router)
npm run preview    # serve the production build, also on port 6023
```

---

## 📓 Quick start — the notebooks

Two notebook tracks, both in the by-hand-then-verify style:

- **`playbook/`** — `P00`–`P29`: a complete from-zero ML/DL/NLP/LLM curriculum that flows into…
- **`notebooks/`** — `00`–`13`: the TinyGPT track, which builds, trains, and probes a real GPT.

They run on a Python kernel with the scientific + PyTorch stack. The simplest setup:

```bash
# one-time: install deps and register a Jupyter kernel named "tinygpt"
python -m pip install -r requirements.txt
python -m ipykernel install --user --name tinygpt --display-name "Python (TinyGPT)"

# launch
python -m jupyter lab
```

Then open `playbook/P00_Python_and_NumPy_for_ML.ipynb` and pick the **Python (TinyGPT)** kernel.

> **Note:** a few notebooks show TensorFlow / Keras / HuggingFace snippets as **read-only**
> (skip-execution) cells — they're for reading the API, not running. Everything that executes uses
> only numpy / pandas / scikit-learn / PyTorch / matplotlib.

---

## 🗂️ Project structure

```text
Build-a-Brain/
├── webapp/                 # the interactive site (React + Vite + TS + Three.js)
│   ├── src/lessons/        #   2-D interactives
│   ├── src/three/          #   lazy-loaded 3-D scenes
│   └── src/pages/          #   chapter content (articles + interactive lessons)
├── playbook/               # P00–P29 — the full ML→LLM curriculum notebooks
├── notebooks/              # 00–13 — the TinyGPT (build-a-GPT) track
├── _builders/              # scripts that generate the playbook notebooks
├── STORY_GUIDE.md          # the "Raising a Mind" narrative spine + voice spec
├── PLAYBOOK_ROADMAP.md
├── TINYGPT_LEARNING_ROADMAP.md
└── README.md
```

---

## 🧭 Guiding principle

> Whenever there's a choice between **simplicity** and **performance**, choose simplicity.
> Every operation should be visible, inspectable, and explainable — and every formula should be
> shown three ways: plain words, a worked example by hand, and code that re-computes the numbers.

## 🤝 Contributing

Contributions are very welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). The golden rule: the math
must be **correct** (verify by hand and against a library), and explanations should stay in the
story voice. After webapp changes, run `npm run build`; after notebook changes, execute them so
outputs are embedded.

## 📜 License

[MIT](LICENSE) © 2026 Harishankar Noudu.
