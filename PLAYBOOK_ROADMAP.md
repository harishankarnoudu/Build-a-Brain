# The Complete ML / DL / NLP / LLM Playbook

> A from-zero-to-domain-expert curriculum. It starts at the true prerequisites
> (the math and Python you actually need), builds classical machine learning,
> then deep learning, then NLP, then transformers — at which point it **flows
> directly into the existing TinyGPT notebooks** (`notebooks/00`–`13`) — and
> finishes with modern LLM engineering (LoRA, RLHF, RAG, quantization, serving).

## The one rule (same as Notebook 00)

Every important idea is shown **three ways**:

1. **Plain words** — what it does and *why we want it*.
2. **A tiny worked example BY HAND** — real small numbers, every multiply/add written out.
3. **Code that re-computes the same numbers** — so the hand answer is *verified*.

Numbers stay tiny (2–3 element vectors) so you can follow every step with a pen.
We prioritise **understanding over performance** — every operation stays visible.

## Libraries

- **Runnable cells** use the installed stack: `numpy, pandas, matplotlib, seaborn,
  scipy, scikit-learn, torch (CPU)`. Everything from-scratch first, library second.
- **TensorFlow / Keras** and **HuggingFace** (`transformers`, `peft`/LoRA, `trl`/RLHF)
  are blocked by enterprise security on this machine, so their code appears as
  **annotated, read-only** cells (clearly marked) shown *side-by-side* with the
  from-scratch version. You learn the API and the idea; the math you can run.

---

## Curriculum

### Part 0 — Foundations (the missing prerequisites)
| # | Notebook | You will be able to… |
|---|----------|----------------------|
| P00 | Python & NumPy for ML | vectorise, broadcast, index, reshape — the language of every later cell |
| P01 | Linear Algebra by hand | dot products, matmul, norms, transpose — *the* operations of a neural net |
| P02 | Calculus & the Chain Rule | derivatives, partials, gradients — the fuel for backprop |
| P03 | Probability, Statistics & Information Theory | distributions, Bayes, expectation, **entropy → cross-entropy** |
| P04 | Data Wrangling & Visualisation | load/clean/inspect data with pandas; read a plot |

### Part 1 — Classical Machine Learning
| # | Notebook | You will be able to… |
|---|----------|----------------------|
| P05 | The ML Mindset | frame a problem: features/labels, train/val/test, overfitting, the learning loop |
| P06 | Linear Regression from scratch | fit a line with gradient descent; verify against `sklearn` |
| P07 | Logistic Regression & Classification | sigmoid, decision boundary, classification cross-entropy |
| P08 | Evaluation, Regularization & Bias–Variance | precision/recall/ROC, L1/L2, why models over/under-fit |
| P09 | Core Algorithms Tour | kNN, Decision Trees, Random Forests, SVM, Naive Bayes, k-Means, PCA |

### Part 2 — Deep Learning
| # | Notebook | You will be able to… |
|---|----------|----------------------|
| P10 | Neurons & Neural Networks from scratch | build a multi-layer net with only numpy |
| P11 | Backpropagation, fully by hand | derive and hand-compute every gradient in a 2-layer net |
| P12 | Optimizers & the Training Loop | SGD, Momentum, RMSProp, **Adam**, LR schedules |
| P13 | Making Deep Nets Train | init, BatchNorm/LayerNorm, dropout, vanishing/exploding gradients |
| P14 | PyTorch Deep Dive (+ Keras/TF mirror) | tensors, autograd, `nn.Module`, a real training loop |
| P15 | Convolutional Neural Networks | convolution by hand, pooling, why CNNs see images |
| P16 | Recurrent Networks (RNN/LSTM/GRU) | model sequences — and feel *why* they hit a wall (→ transformers) |

### Part 3 — Natural Language Processing
| # | Notebook | You will be able to… |
|---|----------|----------------------|
| P17 | Classical NLP | tokenization, Bag-of-Words, TF-IDF, n-gram language models |
| P18 | Word Embeddings from scratch | word2vec (skip-gram) + GloVe intuition; semantic arithmetic |
| P19 | Seq2Seq & the birth of Attention | encoder/decoder, the alignment problem attention solves |

### Part 4 — Transformers & LLMs  → **hands off to the TinyGPT track**
| # | Notebook | You will be able to… |
|---|----------|----------------------|
| P20 | From Attention to the Transformer (bridge) | connect P19 → `notebooks/00_Introduction` and continue 01–13 |

> After P20, work through the existing **`notebooks/01`–`13`**: Tokenizer →
> Embeddings → Neural-Net Basics → Self-Attention → Multi-Head → Transformer Block
> → Positional Encoding → Build TinyGPT → Train → Generate → Interpretability →
> SecurityGPT → Modern Fine-Tuning. Then return here for Part 5.

### Part 5 — Modern LLM Engineering
| # | Notebook | You will be able to… |
|---|----------|----------------------|
| P21 | Pretraining, Scaling Laws & Data | what "training a foundation model" actually means |
| P22 | Supervised Fine-Tuning & Instruction Tuning | turn a base model into an assistant |
| P23 | PEFT: LoRA / QLoRA / Quantization | the math of low-rank adapters; int8/int4 by hand (+ HF read-only) |
| P24 | RLHF: Reward Models, PPO & DPO | reward modelling + policy optimisation from scratch (+ TRL read-only) |
| P25 | RAG, Tool Use & Agents | retrieval, embeddings search, the agent loop |
| P26 | Prompt Engineering & Decoding | greedy/temperature/top-k/top-p, few-shot, chain-of-thought |
| P27 | Evaluation, Alignment & Safety | benchmarks, judging, jailbreaks, guardrails |

### Part 6 — Deployment / MLOps
| # | Notebook | You will be able to… |
|---|----------|----------------------|
| P28 | Serving & Inference Optimization | KV-cache, batching, quantized inference, vLLM concepts |
| P29 | MLOps | experiment tracking, data/versioning, monitoring, reproducibility |

---

## How notebooks are built

Each notebook is generated by a builder script in `_builders/playbook/build_PNN.py`
(using `nbformat`, raw strings to dodge JSON/LaTeX escaping), then executed with
nbconvert against the `tinygpt` kernel so verified outputs are embedded:

```bash
PY="$LOCALAPPDATA/anaconda3/python.exe"
"$PY" _builders/playbook/build_P00.py          # write the .ipynb
"$PY" -m jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name=tinygpt playbook/P00_*.ipynb
```

A helper, `_builders/playbook/_pbcommon.py`, provides `md()`, `code()`, `readonly()`
(for the non-runnable TF/HF cells), and `build(path, cells)` which writes + executes.

## Status

- [x] Roadmap (this file)
- [x] **Part 0 — Foundations (complete, all executed & verified)**
  - [x] P00 — Python & NumPy for ML
  - [x] P01 — Linear Algebra by hand
  - [x] P02 — Calculus & the Chain Rule
  - [x] P03 — Probability, Statistics & Information Theory
  - [x] P04 — Data Wrangling & Visualisation
- [x] **Part 1 — Classical ML (P05–P09) — complete, executed & verified**
  - [x] P05 ML Mindset · P06 Linear Regression · P07 Logistic Regression · P08 Evaluation/Regularization · P09 Algorithms Tour
- [x] **Part 2 — Deep Learning (P10–P16) — complete, executed & verified**
  - [x] P10 Neural Nets · P11 Backprop (grad-check 7e-11, XOR solved) · P12 Optimizers · P13 Training Deep Nets · P14 PyTorch (+Keras read-only) · P15 CNNs · P16 RNNs
- [x] **Part 3 — NLP (P17–P19) — complete, executed & verified**
  - [x] P17 Classical NLP · P18 Word Embeddings (co-occurrence + skip-gram) · P19 Seq2Seq & Attention
- [x] **Part 4 — Transformer bridge (P20) — complete** → routes into TinyGPT notebooks 00–13
- [x] **Part 5 — Modern LLM Engineering (P21–P27) — complete, executed & verified**
  - [x] P21 Pretraining/Scaling · P22 SFT · P23 LoRA/QLoRA/Quantization · P24 RLHF (reward model + DPO) · P25 RAG/Agents · P26 Prompt/Decoding · P27 Eval/Safety
- [x] **Part 6 — Deployment / MLOps (P28–P29) — complete, executed & verified**
  - [x] P28 Serving & Inference (KV-cache) · P29 MLOps

**🎓 ALL 30 NOTEBOOKS COMPLETE** — P00–P29 built, executed, and outputs verified. Index: `playbook/README.md`.
