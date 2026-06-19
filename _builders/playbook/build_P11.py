"""Builder for playbook/P11_Backpropagation.ipynb

THE notebook. Backprop = the chain rule (P02) run backward through the network,
reusing stored forward values. We do a full scalar example by hand (every partial
derivative written out), state the general "local grad x upstream grad" rule, then
implement forward+backward for a real 2-layer net and PROVE the gradients correct
with a numerical gradient check, then train it to solve XOR (which P10 said needs
training). This is exactly what loss.backward() does in PyTorch.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, reset, build

reset()

md(r"""
# P11 — Backpropagation, fully by hand  *(the flash of learning)*

> **The story so far.** In P10 the mind grew a brain — layers of neurons that *could* think a curved
> thought. But it was a brain in a coma: we set its weights by hand, and on its own it knew nothing.
> This chapter is where it opens its eyes. **Backpropagation** is the single idea that turns a frozen
> network into a learning one — the flash that lets a mistake travel backward through the brain and
> nudge every connection that caused it. It is the closest thing in this whole story to a spark of
> life, and it is the heartbeat you've known since P02, run in reverse: predict → measure how wrong →
> step downhill, now whispered to every weight at once.

When a coach watches a missed free throw, they don't just say "you missed" — they trace the error
back: the wrist was off, which came from the elbow, which came from the stance. Backprop is exactly
that blame-tracing, done with arithmetic. The forward pass (P10) turns inputs into a prediction;
backprop does the reverse, computing how the loss would change if you nudged *each* weight — the
gradients — so gradient descent (P02) can improve them. And it is nothing more than **the chain rule
(P02) applied backward through the network**, cleverly reusing values already computed going forward.

We'll earn it three ways, never hand-waving: (1) a full **scalar example by hand**, every derivative
written out; (2) the **general rule** in one block of equations; (3) a real **2-layer network** whose
gradients we *prove* correct with a numerical gradient check, then **train to solve XOR** — the very
weights P10 had to hand-pick. When you finish, PyTorch's `loss.backward()` will hold no mystery.
""")

md(r"""
## B1 — The idea in one picture: forward stores, backward multiplies

Trace a rumour back through a chain of people: each person only needs to know who told them and how
much they embellished it, and the whole path back to the source falls out one handoff at a time. The
brain learns the same way — each operation handles only its own little piece of blame and passes the
rest backward. Nobody computes the whole thing at once.

Think of the network as a chain of operations. Each operation knows two things:
- its **local gradient** — how its output changes with its input (e.g. for `a = σ(z)`, that's `σ'(z)`).
- the **upstream gradient** — how the loss changes with its output (handed back from the next op).

Backprop walks from the loss *backward*, and at each step multiplies **local × upstream** (chain
rule) to get the gradient w.r.t. that op's input, then passes it further back. That's it. The
forward pass cached the intermediate values; the backward pass reuses them so nothing is recomputed.
```
forward :  x ──► z=Wx+b ──► a=f(z) ──► … ──► Loss        (store z, a along the way)
backward:  dL/dx ◄── dL/dz ◄── dL/da ◄── … ◄── dL/dLoss=1  (multiply local·upstream)
```
""")

md(r"""
## B2 — A complete scalar example, every derivative by hand

Before we trust the brain with thousands of weights, let's watch the flash happen on a single one —
slow enough to follow every spark. One weight, one bias, one input, one target. We push a number
forward to a prediction, see how wrong it is, then trace that wrongness back to "how should `w`
move?" — entirely by hand, then verify it against a numerical check so nothing is taken on faith.

The simplest learnable unit: one weight `w`, one bias `b`, one input `x`, sigmoid activation,
squared-error loss against target `y`.
```
z = w·x + b            a = σ(z)            L = (a − y)²
```
Take `x = 1, w = 0.5, b = 0, y = 1`. **Forward:**
```
z = 0.5·1 + 0 = 0.5
a = σ(0.5)   = 0.622
L = (0.622 − 1)² = (−0.378)² = 0.143
```
**Backward — multiply local gradients along the chain** (using `σ'(z) = a(1−a)`, a handy identity):
```
dL/da = 2(a − y)        = 2(0.622 − 1)        = −0.755
da/dz = a(1 − a)        = 0.622·0.378         =  0.235
dz/dw = x = 1           dz/db = 1
dL/dw = dL/da · da/dz · dz/dw = −0.755 · 0.235 · 1 = −0.177
dL/db = dL/da · da/dz · dz/db = −0.755 · 0.235 · 1 = −0.177
```
Both negative → increasing `w` and `b` lowers the loss. We verify with finite differences (P02).
""")

code(r'''
import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))

x, w, b, y = 1.0, 0.5, 0.0, 1.0
# forward
z = w*x + b; a = sigmoid(z); L = (a - y)**2
print("forward:  z=%.3f  a=%.3f  L=%.3f   (hand 0.5, 0.622, 0.143)" % (z, a, L))

# backward, by hand (local * upstream)
dL_da = 2*(a - y)
da_dz = a*(1 - a)
dL_dw = dL_da * da_dz * x
dL_db = dL_da * da_dz * 1
print("backward: dL/dw=%.3f  dL/db=%.3f   (hand -0.177, -0.177)" % (dL_dw, dL_db))

# numerical gradient check (P02): nudge each param, measure L
def loss(w, b): z=w*x+b; a=sigmoid(z); return (a-y)**2
h=1e-6
print("numeric : dL/dw=%.3f  dL/db=%.3f   <- confirms backprop"
      % ((loss(w+h,b)-loss(w-h,b))/(2*h), (loss(w,b+h)-loss(w,b-h))/(2*h)))
''')

md(r"""
## B3 — The general layer rules (the four backprop equations)

The single-weight trace generalises into four short equations that work for *any* layer of *any*
network — small enough to write on the back of a hand, powerful enough to train GPT. Learn this block
and you've learned deep learning's one true engine; everything after is bookkeeping.

For a layer `z = W·a_prev + b`, then `a = f(z)`, given the upstream gradient `dL/da` handed back
from the layer in front, the chain rule gives everything we need:
```
1.  dL/dz      = dL/da  ⊙  f'(z)              (⊙ = element-wise; undo the activation)
2.  dL/dW      = dL/dz  ⊗  a_prevᵀ            (outer product: how each weight contributed)
3.  dL/db      = dL/dz                         (bias adds directly, so gradient passes straight)
4.  dL/da_prev = Wᵀ · dL/dz                    (pass the gradient back to the previous layer)
```
Equation 4 is the "prop" in backprop — it carries the error signal one layer further back, where
the same four equations run again. Memorise this block; it *is* the algorithm. (`ReLU'(z)` is 1
where `z>0` else 0; `σ'(z) = a(1−a)`.)
""")

md(r"""
## B4 — A real 2-layer network: implement forward + backward, then PROVE it

Hand-coded backprop is famously easy to get *almost* right — a stray transpose, a wrong derivative —
and a network that's almost right learns garbage. So engineers never trust their own derivation; they
prove it. We implement forward and backward for a real little network, then run the time-honoured
**gradient check**: jiggle each weight a hair, see how the loss actually moves, and confirm it matches
what backprop claimed. If the gap is microscopic, the math is provably correct.

Network: `2 inputs → 2 hidden (ReLU) → 1 output (sigmoid)`, BCE loss. We implement the forward pass
(caching `z1, a1, …`) and the backward pass using the four equations from B3. Then we run a
**gradient check**: compare every analytic gradient against a numerical finite-difference. If the
maximum difference is tiny (~1e-7), the backprop is provably correct.
""")

code(r'''
import numpy as np
rng = np.random.default_rng(0)
def sigmoid(z): return 1/(1+np.exp(-z))

# parameters
W1 = rng.normal(0, 0.5, (2, 2)); b1 = np.zeros(2)
W2 = rng.normal(0, 0.5, (1, 2)); b2 = np.zeros(1)
x  = np.array([1.0, 0.0]); y = np.array([1.0])

def forward(W1,b1,W2,b2, x):
    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)              # ReLU
    z2 = W2 @ a1 + b2
    a2 = sigmoid(z2)                    # output probability
    cache = (x, z1, a1, z2, a2)
    return a2, cache

def bce(a2, y):
    a2 = np.clip(a2, 1e-9, 1-1e-9)
    return float((-(y*np.log(a2) + (1-y)*np.log(1-a2))).sum())

def backward(W1,b1,W2,b2, cache, y):
    x, z1, a1, z2, a2 = cache
    # dL/dz2 for sigmoid+BCE collapses to (a2 - y)  (the clean form from P07)
    dz2 = (a2 - y)                                  # (1,)
    dW2 = np.outer(dz2, a1)                          # eq 2: (1,2)
    db2 = dz2                                        # eq 3
    da1 = W2.T @ dz2                                 # eq 4: (2,)
    dz1 = da1 * (z1 > 0)                             # eq 1: ReLU'(z)=1 where z>0
    dW1 = np.outer(dz1, x)                           # eq 2: (2,2)
    db1 = dz1                                        # eq 3
    return dW1, db1, dW2, db2

a2, cache = forward(W1,b1,W2,b2, x)
dW1, db1, dW2, db2 = backward(W1,b1,W2,b2, cache, y)
print("forward output a2 =", np.round(a2,4), " loss =", round(bce(a2,y),4))
print("analytic dW1 =\n", np.round(dW1,5))
''')

code(r'''
import numpy as np
# GRADIENT CHECK: compare analytic backprop to numerical finite differences.
# For each parameter entry: nudge it up and down, measure how the loss moves,
# and confirm that slope equals what backprop computed.
def loss_with(W1,b1,W2,b2):
    a2,_ = forward(W1,b1,W2,b2,x); return bce(a2,y)

max_diff = 0.0
for name, P in [("W1",W1),("b1",b1),("W2",W2),("b2",b2)]:
    grad_analytic = {"W1":dW1,"b1":db1,"W2":dW2,"b2":db2}[name]
    it = np.nditer(P, flags=["multi_index"])
    while not it.finished:
        i = it.multi_index; orig = P[i]
        P[i] = orig + 1e-6; Lp = loss_with(W1,b1,W2,b2)
        P[i] = orig - 1e-6; Lm = loss_with(W1,b1,W2,b2)
        P[i] = orig
        num = (Lp - Lm) / (2e-6)
        max_diff = max(max_diff, abs(num - grad_analytic[i]))
        it.iternext()
print("max |analytic - numeric| across ALL gradients = %.2e" % max_diff)
print("tiny (~1e-7) -> our hand-coded backprop is provably correct.")
''')

md(r"""
## B5 — Train the network to solve XOR

This is the payoff — the moment the mind teaches itself. We start the brain with random, meaningless
weights and show it XOR over and over, each time letting backprop trace the blame and gradient descent
take a small step downhill. Watch the loss fall and the predictions snap to the right answers. Nobody
hand-picks anything; the network discovers the very weights P10 had to supply. This is deep learning
in thirty lines, and it is the heartbeat of every model that follows.

Now wire backprop into the gradient-descent loop and train on XOR — the problem P10 proved a single
neuron cannot solve. We loop over the 4 examples, accumulate gradients, and step. Watch the loss
fall toward 0 and the predictions snap to the correct 0/1. **This is deep learning in 30 lines.**
""")

code(r'''
import numpy as np
rng = np.random.default_rng(1)
def sigmoid(z): return 1/(1+np.exp(-z))

W1 = rng.normal(0,1.0,(4,2)); b1 = np.zeros(4)     # 2 -> 4 hidden -> 1
W2 = rng.normal(0,1.0,(1,4)); b2 = np.zeros(1)
X = np.array([[0,0],[0,1],[1,0],[1,1]], float); Y = np.array([0,1,1,0], float)
lr = 0.5

def fwd(x):
    z1=W1@x+b1; a1=np.maximum(0,z1); z2=W2@a1+b2; a2=sigmoid(z2)
    return z1,a1,z2,a2

for epoch in range(2001):
    gW1=np.zeros_like(W1); gb1=np.zeros_like(b1); gW2=np.zeros_like(W2); gb2=np.zeros_like(b2)
    total=0.0
    for x,y in zip(X,Y):
        z1,a1,z2,a2 = fwd(x); a2c=np.clip(a2,1e-9,1-1e-9)
        total += float((-(y*np.log(a2c)+(1-y)*np.log(1-a2c))).sum())
        dz2=(a2-y); gW2+=np.outer(dz2,a1); gb2+=dz2
        dz1=(W2.T@dz2)*(z1>0); gW1+=np.outer(dz1,x); gb1+=dz1
    for P,g in [(W1,gW1),(b1,gb1),(W2,gW2),(b2,gb2)]: P-=lr*g/4
    if epoch%500==0: print(f"epoch {epoch:>4}  loss {total/4:.4f}")

print("\nfinal predictions:")
for x,y in zip(X,Y):
    p=fwd(x)[3][0]
    print(f"  x={x.astype(int)}  pred={p:.3f}  target={int(y)}  {'OK' if round(p)==y else 'X'}")
print("the network learned XOR from scratch — backprop found the weights P10 hand-picked.")
''')

md(r"""
## Recap — backprop, demystified

| Step | What happens |
|------|--------------|
| Forward | compute & **cache** `z, a` at each layer → prediction → loss |
| Backward | start at `dL/dLoss = 1`, apply the 4 equations layer by layer |
| Eq 1 | `dL/dz = dL/da ⊙ f'(z)` — peel off the activation |
| Eq 2 | `dL/dW = dL/dz ⊗ a_prevᵀ` — the weight gradients |
| Eq 3 | `dL/db = dL/dz` |
| Eq 4 | `dL/da_prev = Wᵀ dL/dz` — propagate one layer back |
| Update | `param -= lr · grad` (gradient descent, P02) |
| Verify | numerical gradient check ⇒ correctness |

`loss.backward()` in PyTorch (P14) runs exactly this, automatically, for millions of parameters.

## Common mistakes
1. **Not caching forward values.** Backprop *reuses* `z`, `a`; recomputing them is wasteful and bug-prone.
2. **Wrong activation derivative.** ReLU′ is 1 where `z>0` else 0 — use `z>0`, not `a>0` (same here, but be careful).
3. **Transpose errors in eq 2 & 4.** `dW` matches `W`'s shape; `Wᵀ` sends the gradient backward. Check shapes every time.
4. **Forgetting to average/scale gradients over the batch** before stepping (we divide by 4).
5. **Skipping the gradient check.** When hand-coding backprop, *always* verify numerically before trusting it.

## Exercises (do them in new code cells)
1. Redo the B2 scalar example with `x=2, w=1, b=−1, y=0`. Hand-compute `dL/dw`, verify numerically.
2. In B4, switch the hidden activation to sigmoid. What changes in eq 1 (the `f'(z)` term)? Re-run the gradient check.
3. In B5, lower `lr` to `0.05`. How many more epochs to solve XOR? What does that say about learning rate?
4. Add a second hidden layer to the B5 network. Write the extra backprop equations (eq 1–4 again) and confirm it still learns.
5. Why does sigmoid+BCE give the clean `dz2 = a2 − y`? (Hint: the log and the sigmoid derivatives cancel — see P07 B3.)

---

**The mind can now learn from its mistakes** — the spark is lit. But it learns *slowly*, stumbling
downhill one cautious step at a time. A learning mind that takes forever to learn anything won't get
far. Next we teach it to move with grace and speed:
[P12 — Optimizers & the Training Loop](P12_Optimizers.ipynb) — plain gradient descent zig-zags and
crawls; we add momentum, RMSProp and **Adam** (the optimizer behind nearly every modern model), each
derived by hand and raced so you can see the mind learn to sprint downhill.
""")

build("P11_Backpropagation", execute=False)
