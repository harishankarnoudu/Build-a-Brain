"""Builder for playbook/P29_MLOps.ipynb

The final notebook. Turning models into reliable products: experiment tracking,
data/model versioning, reproducibility (seeds), the deployment lifecycle, and
monitoring for drift. Mostly practice + a runnable reproducibility/seed demo and a
tiny drift-detection demo. Closes the playbook with a capstone map.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P29 — MLOps  *(keep the mind alive, healthy and honest — the loop never closes)*

> **The story so far.** The mind is grown and out in the world, fast enough to help millions
> (P22–P28). You might think the story is over. It isn't — it's *circular*. A mind let loose and then
> forgotten doesn't stay good; the world keeps changing under it, and a mind frozen in last year's
> reality slowly drifts into being wrong. Raising a mind was never a finish line. It's a loop you
> commit to: watch it, remember what you did, catch it when reality shifts, and quietly teach it
> again. This final chapter is about keeping the mind you raised alive, healthy, and honest — forever.

A model in a notebook is an experiment; a model people depend on is a **product** — a mind you've
promised to look after. **MLOps** is the discipline that keeps that promise: tracking experiments
(so you never lose what worked), versioning data and models, ensuring reproducibility, deploying
safely, and **monitoring** for the day reality drifts away from the world the mind was raised in. This
closing notebook gives you the practices and a couple of runnable demos (reproducibility and drift
detection), then maps the whole journey you just finished.
""")

md(r"""
## B1 — Experiment tracking: never lose a result again

Every parent keeps a baby book — what worked, what didn't, the day it first did the thing. You'll
raise the mind hundreds of times over (different LRs, model sizes, data), and without notes you'll be
unable to answer the only question that matters six months later: "which run produced the best mind,
and with what settings?" **Experiment tracking** logs
every run's **config + metrics + artifacts** so results are comparable and reproducible. Tools:
MLflow, Weights & Biases, TensorBoard. The idea is simple — log a record per run:
""")

code(r'''
# a minimal experiment tracker: log config + final metric for each run, then compare
runs = []
def log_run(name, config, metric):
    runs.append({"name": name, **config, "val_loss": metric})

log_run("baseline",   {"lr": 1e-3, "layers": 4}, 2.10)
log_run("more_layers",{"lr": 1e-3, "layers": 8}, 1.85)
log_run("higher_lr",  {"lr": 3e-3, "layers": 8}, 1.72)
log_run("too_high_lr",{"lr": 1e-1, "layers": 8}, 9.99)   # diverged

best = min(runs, key=lambda r: r["val_loss"])
print("all runs:")
for r in runs: print("  ", r)
print("\nbest run:", best["name"], "-> val_loss", best["val_loss"], "with lr", best["lr"], "layers", best["layers"])
print("real trackers (MLflow/W&B) do this with curves, artifacts, and a UI — but this is the essence.")
''')

md(r"""
## B2 — Reproducibility: same inputs → same outputs

If you can't raise the *same* mind twice from the same recipe, you don't have a recipe — you have luck.
Science requires reproducibility, and so does debugging. The big levers: **set random seeds**, pin
**library versions** (`requirements.txt`), and **version your data**. Without a seed, two "identical"
training runs differ because of random init (P13) and shuffling. We prove the seed fixes it.
""")

code(r'''
import numpy as np
def run(seed=None):
    rng = np.random.default_rng(seed)
    return rng.normal(size=3).round(3)

print("no fixed seed (differs each call):")
print("  ", run(), "vs", run())
print("fixed seed (identical every time):")
print("  ", run(42), "vs", run(42))
print("\nset seeds for numpy, torch, AND python's random; pin versions; version data -> reproducible runs.")
''')

md(r"""
## B3 — The deployment lifecycle: data & model versioning

You wouldn't push a new version of the mind to everyone at once any more than you'd hand a teenager
the car keys on day one — you let it drive a little, watch closely, and keep the ability to take the
keys back instantly. A model is a *function of its data and code*, so all three are versioned together:
```
data version  +  code/config version  +  model weights version  ->  a reproducible release
```
The lifecycle: **train → validate → stage → canary (serve to a small % of traffic) → full rollout →
monitor**, with the ability to **roll back** to a previous model version instantly if something goes
wrong. Treating models like software releases (not one-off artifacts) is the heart of MLOps.
""")

readonly(r'''
# What this looks like with real tools (read-only sketch):
import mlflow
with mlflow.start_run():
    mlflow.log_params({"lr": 3e-3, "layers": 8})     # config (B1)
    mlflow.log_metric("val_loss", 1.72)              # metrics
    mlflow.log_artifact("model.pt")                  # the versioned weights
# data versioning: tools like DVC track datasets the way git tracks code.
# deployment: a model registry promotes versions  staging -> production, with rollback.
''')

md(r"""
## B4 — Monitoring: catch drift before users do

This is the part that makes the loop never close. The world the mind grew up in keeps moving — new
slang, new events, new ways people phrase things — and a mind that doesn't notice quietly grows
out-of-touch, like a wise elder still giving advice for a world that no longer exists. A deployed
model degrades silently when the live data **drifts** away from training data (new slang, new attack
patterns, seasonal shifts). You can't see this from accuracy alone (you often lack live
labels), so you monitor the *input distribution* and outputs. A simple detector: compare the live
feature mean to the training mean — a large gap signals drift and triggers a retrain.
""")

code(r'''
import numpy as np
rng = np.random.default_rng(0)
train_feature = rng.normal(50, 5, size=1000)        # training data: mean ~50

def drift_score(live):
    return abs(live.mean() - train_feature.mean()) / train_feature.std()

normal_live = rng.normal(50, 5, size=200)            # same distribution
drifted_live = rng.normal(62, 5, size=200)           # shifted! (e.g. new user behaviour)
print(f"drift score (normal traffic) : {drift_score(normal_live):.2f}  -> OK")
print(f"drift score (shifted traffic): {drift_score(drifted_live):.2f}  -> ALERT, retrain")
print("\nmonitor inputs/outputs continuously; alert on drift; retrain on fresh data. The loop never ends.")
''')

md(r"""
## Recap — MLOps

| Practice | Why | Tool |
|----------|-----|------|
| Experiment tracking | compare/repro runs | MLflow, W&B, TensorBoard |
| Reproducibility | same in → same out | seeds, pinned versions, DVC |
| Versioning | data+code+weights = a release | git, DVC, model registry |
| Deployment lifecycle | safe rollout + rollback | canary, registries |
| Monitoring | catch drift early | input/output dashboards, alerts |

## Common mistakes
1. **No experiment tracking.** You can't reproduce or beat a result you didn't log.
2. **Forgetting seeds** (and seeding only numpy, not torch/random) → "irreproducible" runs.
3. **Versioning code but not data.** The model depends on both; pin them together.
4. **Deploying with no monitoring.** Silent drift degrades quality until users complain.
5. **No rollback plan.** Always be able to revert to the last known-good model instantly.

## Exercises (do them in new code cells)
1. Add two more runs to B1's tracker (vary `layers`/`lr`). Does the best run change?
2. Seed `torch.manual_seed(0)` and generate `torch.randn(3)` twice. Are they identical?
3. In B4, at what live mean does the drift score exceed 2.0? Solve by hand, then check.
4. List the data/code/model versions you'd record to reproduce a result six months later.
5. Why can accuracy look fine while the model is silently failing on a drifted subgroup?

---

# 🎓 The journey comes full circle — the mind you raised is out in the world 🎉

Look back at the life you just watched unfold. It began, on day one, with a newborn mind opening its
eyes and seeing nothing but **numbers** (P00). It ends with that same mind grown up — fluent, tasteful,
honest, and out in the world helping millions — and *still learning*, because the loop you just closed
in this chapter never really closes. From a single bare number to a thinking machine. Here is every
stage of that life:

- **Part 0 — Foundations** *(born)*: NumPy, linear algebra, calculus, probability/information, data.
- **Part 1 — Classical ML** *(first instincts)*: the mindset, regression, classification, evaluation, the algorithm zoo.
- **Part 2 — Deep Learning** *(a brain forms)*: neurons, **backprop by hand**, optimizers, training deep nets, PyTorch, CNNs, RNNs.
- **Part 3 — NLP** *(learning to read & mean)*: classical text, embeddings, seq2seq & attention.
- **Part 4 — Transformers** *(its true form)*: the bridge into **TinyGPT (`notebooks/00`–`13`)** — built, trained, probed.
- **Part 5 — LLM Engineering** *(raising a real mind)*: pretraining & scaling, SFT, **LoRA/QLoRA**, **RLHF/DPO**, RAG/agents, decoding, eval/safety.
- **Part 6 — Deployment** *(out into the world)*: serving it to millions, and keeping it alive, healthy, and honest.

And the same heartbeat beat under every stage — **predict → measure how wrong → step downhill** — from
your first line of regression to pretraining on the whole internet: *the same loop, a billion-fold
bigger*. The mind's single primitive sense, the **dot product** — "how much do two things agree?" —
returned again and again, in neurons, in attention, and finally in the library that lets the grown
mind look things up.

Every concept was shown three ways — plain words, a tiny worked example **by hand**, and code that
**verified** the numbers. You didn't just read about machine learning; you re-derived it. That
first-principles understanding is exactly what makes a domain expert.

**Where to go next:** the mind is yours now — go raise one of your own. Pick a real dataset or model
and build something end-to-end: fine-tune an open model with LoRA (P23), stand up a RAG system (P25),
or extend TinyGPT. You started with a single number. You can build a mind. 🎉
""")

build("P29_MLOps", execute=False)
