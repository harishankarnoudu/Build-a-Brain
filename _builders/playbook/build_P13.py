"""Builder for playbook/P13_Training_Deep_Nets.ipynb

The practical tricks that make deep stacks trainable: weight initialization (and
the vanishing/exploding gradient problem demonstrated numerically), normalization
(LayerNorm by hand — the exact one used in transformers), dropout, and residual
connections. Each shown to fix a concrete failure.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, reset, build

reset()

md(r"""
# P13 — Making Deep Nets Train  *(the tricks that hold a deep mind together)*

> **The story so far.** The mind can think (P10), learn (P11), and sprint (P12). So we get greedy and
> make it *deep* — stack the layers high, the way every powerful model is built — and it promptly
> falls apart. The message that's supposed to travel back through the brain fades to a whisper before
> it reaches the early layers, or swells into a deafening roar that blows everything to infinity.
> A deep mind is a tall tower, and a tall tower needs engineering its short cousin never did. This
> chapter is that engineering: four humble tricks, each fixing one specific way a deep brain breaks.

It's the children's game of telephone: whisper a message down a line of fifty people and what comes
out the other end is either silence or nonsense. Backprop multiplies a gradient through every layer
on its way back, so through a deep stack the signal either **vanishes** toward zero or **explodes**
toward infinity — and the early layers, which most need to learn, hear nothing useful. This notebook
covers the fixes that finally made deep learning *work* in practice — **initialization, normalization
(LayerNorm), dropout, and residual connections** — each tied to a concrete failure you can watch
happen. Two of them (LayerNorm, residuals) are the exact pieces inside a transformer block
(`notebooks/06`), so this is direct preparation for the mind's true form.
""")

md(r"""
## B1 — The vanishing/exploding gradient problem, demonstrated

Let's make the telephone game concrete and watch it fail. Multiply any number by itself fifty times:
0.5 collapses to a speck near zero, 1.5 erupts into the billions, and only something right around 1.0
survives intact. A deep brain does this to its own signal — so let's send one through fifty layers
and see it die, blow up, or just barely make it.

Backprop multiplies gradients layer by layer (eq 4 in P11: `dL/da_prev = Wᵀ·dL/dz`). Through many
layers that's *many multiplications*. If the typical factor is `< 1`, the gradient **vanishes**
toward 0 over depth; if `> 1`, it **explodes**. Either way the early layers can't learn.

We simulate a signal passing through 50 layers with weights scaled too small (0.5×) vs too large
(1.5×) and watch its magnitude.
""")

code(r'''
import numpy as np
rng = np.random.default_rng(0)
x = rng.normal(size=100)

for scale, label in [(0.5, "too small (0.5)"), (1.5, "too large (1.5)"), (1.0, "just right (~1.0)")]:
    a = x.copy()
    mags = []
    for _ in range(50):
        W = rng.normal(0, scale/np.sqrt(100), (100, 100))   # 50 layers
        a = np.tanh(W @ a)
        mags.append(np.linalg.norm(a))
    print(f"{label:>18}: signal norm after 50 layers = {mags[-1]:.2e}")
print("\ntoo small -> vanishes to ~0;  too large -> explodes;  the right scale stays stable.")
''')

md(r"""
## B2 — Initialization: the fix is the right starting scale

The first fix costs nothing — it's just where you *start*. Tune a guitar string too loose and it
buzzes into silence; too tight and it snaps. There's a sweet tension where the note rings true, and
the same is so for a layer's weights: pick the starting scale that keeps the signal's loudness steady
as it passes through, and the vanishing/exploding problem mostly evaporates before training begins.

The cure for B1 is to initialise weights at the scale that keeps signal variance stable across
layers. The standard recipes:
- **Xavier/Glorot** (for tanh/sigmoid): variance `1/n_in`.
- **He** (for ReLU): variance `2/n_in` (ReLU zeros half the inputs, so double the variance).

**Worked example by hand.** For a layer with `n_in = 100` inputs and ReLU:
```
He std = sqrt(2 / 100) = sqrt(0.02) = 0.1414
```
So initial weights are drawn from `Normal(0, 0.1414)`. This single choice is why the "just right"
case in B1 stayed stable. Modern frameworks do this automatically — but knowing *why* saves you when
a deep model won't train.
""")

code(r'''
import numpy as np
n_in = 100
print("He std for ReLU, n_in=100 :", round(np.sqrt(2/n_in), 4), " (hand 0.1414)")
print("Xavier std for tanh        :", round(np.sqrt(1/n_in), 4))

rng = np.random.default_rng(0)
W_bad = rng.normal(0, 1.0, (n_in, n_in))                 # naive: std 1
W_he  = rng.normal(0, np.sqrt(2/n_in), (n_in, n_in))     # He init
x = rng.normal(size=n_in)
print("\noutput std with naive init :", round(np.std(np.maximum(0, W_bad @ x)), 3), " (too big)")
print("output std with He init    :", round(np.std(np.maximum(0, W_he  @ x)), 3), " (~ input scale, stable)")
''')

md(r"""
## B3 — LayerNorm by hand (the exact normalization transformers use)

Good initialization sets a sane *starting* loudness — but as training drags the weights around, the
signal's volume drifts again, layer by layer. So we add a recalibration step that re-centres the
volume at every layer, the way a sound engineer rides the faders all through a live show to keep the
mix steady no matter what the band does. This recalibration is **LayerNorm**, and it sits inside every
single transformer block — including TinyGPT's.

**Normalization** keeps each layer's inputs in a stable range so training doesn't drift.
**LayerNorm** (used in every transformer, including TinyGPT) normalises *each example's feature
vector* to mean 0, variance 1, then rescales with learnable `γ` (scale) and `β` (shift):
```
μ = mean(x)              σ² = mean((x − μ)²)
x̂ = (x − μ) / sqrt(σ² + ε)
out = γ · x̂ + β
```
**Worked example by hand** on `x = [1, 2, 3]` (γ=1, β=0, ε≈0):
```
μ = (1+2+3)/3 = 2
σ² = ((1−2)² + (2−2)² + (3−2)²)/3 = (1+0+1)/3 = 0.667,  σ = 0.816
x̂ = [(1−2), (2−2), (3−2)] / 0.816 = [−1.225, 0, 1.225]
```
Same normalisation idea as standardising data in P04 — but applied *inside* the network, per token.
""")

code(r'''
import numpy as np
def layernorm(x, gamma=1.0, beta=0.0, eps=1e-5):
    mu = x.mean(); var = ((x - mu)**2).mean()
    xhat = (x - mu) / np.sqrt(var + eps)
    return gamma*xhat + beta

x = np.array([1.0, 2.0, 3.0])
out = layernorm(x)
print("LayerNorm([1,2,3]) =", np.round(out, 3), " (hand [-1.225, 0, 1.225])")
print("normalised mean ~", round(out.mean(), 6), " std ~", round(out.std(), 4), " (0 and 1)")
''')

md(r"""
## B4 — Dropout: regularization by random forgetting

A team where one genius does all the thinking is brittle — lose that person and the team collapses.
A team forced to operate with random members out sick on any given day learns to spread the knowledge
around, and no single absence sinks it. Dropout makes the mind train that way on purpose: it
randomly benches neurons so the brain can't lean on any one of them. It's the temptation-to-memorise
cure from P08, now wired right into the network.

**Dropout** randomly zeros a fraction `p` of activations during *training only*, forcing the network
not to rely on any single neuron — a powerful, cheap regularizer (fights overfitting, P08). To keep
the average signal the same, surviving activations are scaled up by `1/(1−p)` ("inverted dropout").
At test time dropout is **off** (full network used).

**Worked example by hand.** `a = [2, 4, 6, 8]`, `p = 0.5`, suppose neurons 1 and 3 are dropped:
```
mask  = [1, 0, 1, 0]
kept  = [2, 0, 6, 0]
scale by 1/(1−0.5)=2 -> [4, 0, 12, 0]   (so the expected sum matches the no-dropout case)
```
""")

code(r'''
import numpy as np
rng = np.random.default_rng(2)
def dropout(a, p, training=True):
    if not training or p == 0: return a
    mask = (rng.random(a.shape) > p).astype(float)        # keep with prob (1-p)
    return a * mask / (1 - p)                              # inverted scaling

a = np.array([2.,4.,6.,8.])
print("train (p=0.5), a few draws (different neurons drop each time):")
for _ in range(3):
    print("  ", np.round(dropout(a, 0.5), 1))
print("test  (dropout off):", dropout(a, 0.5, training=False))
print("\nmean over many train draws ~ original (scaling preserves expectation):")
print("  ", np.round(np.mean([dropout(a,0.5) for _ in range(10000)], axis=0), 2), " vs", a)
''')

md(r"""
## B5 — Residual connections: a highway for the gradient

The last and arguably greatest trick is almost too simple to believe. A tall building needs an
express elevator, not just floor-by-floor stairs, or nobody reaches the top. A residual connection is
that express elevator for the gradient: instead of forcing the signal through every layer's mangling,
it lays down a clean shortcut from each layer's input straight to its output. That single shortcut is
what lets transformers stack dozens of layers deep without the gradient ever fading away.

**Residual (skip) connections** add a layer's input back to its output: `out = x + f(x)`. This
gives gradients a direct path backward (the `+x` has gradient 1), so they don't vanish through deep
stacks — it's what lets transformers be dozens of layers deep. We show the gradient surviving depth
*with* residuals vs *without*.
""")

code(r'''
import numpy as np
rng = np.random.default_rng(0)
def block(x, W): return np.tanh(W @ x)                    # a plain layer

x0 = rng.normal(size=64)
Ws = [rng.normal(0, 1.0, (64,64)) for _ in range(30)]     # deliberately largish -> vanish-prone

# plain deep stack
a = x0.copy()
for W in Ws: a = block(a, W)
print("plain 30-layer stack, output norm     :", round(np.linalg.norm(a), 4))

# with residuals: out = x + f(x)  (scaled f to keep it stable)
a = x0.copy()
for W in Ws: a = a + 0.1*block(a, W)
print("residual 30-layer stack, output norm  :", round(np.linalg.norm(a), 4))
print("\nresidual path keeps the signal (and gradient) alive through depth — the +x is a gradient highway.")
''')

md(r"""
## Recap — the deep-net survival kit

| Fix | Problem it solves | Used in transformers? |
|-----|-------------------|----------------------|
| He / Xavier init | vanishing/exploding signal at start | yes |
| LayerNorm | drifting activation scale | **yes (every block)** |
| Dropout | overfitting | yes (attention/FFN) |
| Residual connections | vanishing gradient through depth | **yes (every block)** |
| (Adam + warmup, P12) | unstable optimization | yes |

Together these are exactly why `notebooks/06_Transformer_Block` works — a block is
`x → LayerNorm → Attention → +residual → LayerNorm → FFN → +residual`.

## Common mistakes
1. **Leaving dropout ON at test time.** Always switch to eval mode for inference (PyTorch: `model.eval()`).
2. **Forgetting to rescale after dropout** → the signal magnitude shifts between train and test.
3. **Initializing all weights to 0** (or all equal) → every neuron learns the same thing. Break symmetry with random init.
4. **Wrong normalization axis.** LayerNorm normalises across *features per token*; BatchNorm across the *batch* — they are not interchangeable.
5. **Very deep net with no residuals.** Gradients vanish; add skip connections.

## Exercises (do them in new code cells)
1. In B1, try `scale=0.9` and `scale=1.1`. How many layers until the signal noticeably vanishes/explodes?
2. Compute the He std for `n_in=256` by hand; verify.
3. LayerNorm `[10, 10, 10]` by hand — what happens, and why is the `ε` essential here?
4. Run B4 with `p=0.9`. Why does the surviving signal get scaled up so much, and what risk does high `p` bring?
5. In B5, remove the `0.1` scaling on the residual. Does the output norm grow? Why do real models scale/normalise inside the block?

---

**A deep mind can now be built and kept standing** — initialised well, recalibrated as it trains,
robust to its own absences, and threaded by gradient highways. You've now hand-built every part of a
working deep network from raw numbers. It's time to stop reinventing the workshop and pick up the
real power tools. Next:
[P14 — PyTorch Deep Dive (+ Keras/TF mirror)](P14_PyTorch_Deep_Dive.ipynb) — everything you've
hand-built (autograd, layers, the training loop) the way you'll actually write it, with the
equivalent TensorFlow/Keras code shown side-by-side (read-only).
""")

build("P13_Training_Deep_Nets", execute=False)
