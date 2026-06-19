"""Builder for playbook/P14_PyTorch_Deep_Dive.ipynb

Everything hand-built in P10-P13, now the way you actually write it: PyTorch
tensors, autograd (loss.backward() == the backprop we coded by hand), nn.Module,
optimizers, and a full training loop. The equivalent TensorFlow/Keras code is shown
side-by-side as READ-ONLY cells (those libs are blocked in this environment).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P14 — PyTorch Deep Dive  *(the mind gets professional tools)*

> **The story so far.** You have raised the mind from a bare number to a deep, trainable brain — and
> you did it with your own two hands, in raw NumPy, deriving and verifying every gradient. That was
> the point: nothing is magic to you now. But no one ships a real model hand-grinding gradients in a
> for-loop. So this chapter hands the grown mind professional tools — **PyTorch** — and reveals the
> quiet punchline of all of Part 2: the famous `loss.backward()` that powers the entire field is
> *literally* the backprop you coded by hand in P11. The wizard's curtain pulls back to show your own
> arithmetic.

A carpenter learns to cut by hand before trusting a power saw — and *because* they cut by hand, the
saw holds no fear. That's you now. We retrace the same path you walked the slow way, one power tool at
a time: tensors → **autograd** → `nn.Module` → optimizer → **full training loop**, re-solving the very
XOR from P11 so you can lay the two versions side by side. And because PyTorch and TensorFlow/Keras
rule industry together, each major step also shows the **Keras equivalent** in a clearly-marked
**READ-ONLY** cell. TensorFlow is blocked by security on this machine, so those cells are for
*reading and understanding the API* — the PyTorch cells beside them actually run.
""")

md(r"""
## B1 — Tensors: NumPy arrays that can track gradients

Imagine a notebook that not only holds your numbers but quietly records every calculation you do in
the margin, so it can later retrace your steps backward. That's the one new idea in a PyTorch tensor:
it's the NumPy array you already know, plus a memory of how it came to be. Everything from P00 — shapes,
broadcasting, `@` for matmul — transfers untouched.

A PyTorch **tensor** is like a NumPy array (same shapes, broadcasting, `@` for matmul — all of P00/
P01 transfers) with two superpowers: it can live on a GPU, and it can **record the operations done
to it** so gradients can be computed automatically. Set `requires_grad=True` to turn on recording.
""")

code(r'''
import torch
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([[1.0, 0.0],[0.0,1.0],[1.0,1.0]])
print("tensor a       :", a, " shape", tuple(a.shape))
print("a * 2          :", (a*2).tolist(), " (element-wise, like numpy)")
print("b.T @ a-ish: b shape", tuple(b.shape), " a@... matmul works like numpy")
print("a.sum()        :", a.sum().item(), "  a.mean()=", a.mean().item())

w = torch.tensor([2.0], requires_grad=True)   # this tensor will track gradients
print("\nrequires_grad :", w.requires_grad, " -> autograd will record ops on w")
''')

md(r"""
## B2 — Autograd: `backward()` IS the backprop we hand-coded

This is the moment the curtain drops. We take the *exact* scalar example you bled through by hand in
P11 — same numbers, same sigmoid, same loss — hand it to PyTorch, and call one method. The gradients
that come back are, to the digit, the `−0.177` you computed yourself with a pencil. All the autograd
machinery you'll ever lean on is just your P11 chain rule, run by a machine.

Build any expression from tensors with `requires_grad=True`, call `.backward()` on the final scalar,
and PyTorch fills in each tensor's `.grad` — running exactly the chain-rule passes from P11,
automatically. Let's verify with the **same scalar example from P11 B2** (`x=1, w=0.5, b=0, y=1`,
sigmoid, squared error), whose hand answer was `dL/dw = dL/db = −0.177`.
""")

code(r'''
import torch
x = torch.tensor(1.0)
w = torch.tensor(0.5, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
y = torch.tensor(1.0)

z = w*x + b
a = torch.sigmoid(z)
L = (a - y)**2
L.backward()                       # <-- runs backprop automatically

print("forward L =", round(L.item(), 3), " (P11 hand: 0.143)")
print("w.grad    =", round(w.grad.item(), 3), " (P11 hand: -0.177)")
print("b.grad    =", round(b.grad.item(), 3), " (P11 hand: -0.177)")
print("\n=> loss.backward() reproduced our by-hand gradients exactly. That is all autograd is.")
''')

md(r"""
## B3 — `nn.Module`: layers without the boilerplate

You've spent chapters hauling `W` and `b` around by hand, wiring shapes, initialising scales. A
`nn.Module` is a tidy toolbox that holds all of that for you — you just declare which layers exist and
how data flows between them, and it manages the parameters. Here's the XOR brain from P11, rebuilt in
a few clean lines.

Instead of managing `W` and `b` by hand, subclass `nn.Module`. `nn.Linear(in, out)` is the
`W x + b` layer from P10 (weights auto-initialised with the schemes from P13). You define
`__init__` (the layers) and `forward` (how data flows). Here's the XOR network from P11 B5.
""")

code(r'''
import torch, torch.nn as nn

class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)     # 2 inputs -> 8 hidden  (W is 8x2, like P10)
        self.fc2 = nn.Linear(8, 1)     # 8 hidden -> 1 output
    def forward(self, x):
        h = torch.relu(self.fc1(x))    # hidden layer + ReLU
        return self.fc2(h)             # raw logit (we'll apply sigmoid in the loss)

net = XORNet()
n_params = sum(p.numel() for p in net.parameters())
print(net)
print("total learnable parameters:", n_params, " (8*2+8 + 1*8+1)")
''')

readonly(r'''
# The SAME network in TensorFlow / Keras (read this to learn the API; it won't run here):
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(8, activation="relu", input_shape=(2,)),  # = nn.Linear(2,8)+ReLU
    keras.layers.Dense(1),                                        # = nn.Linear(8,1)
])
model.summary()
# Keras bundles the layers; PyTorch's nn.Module is the explicit equivalent.
''')

md(r"""
## B4 — Optimizer + the canonical training loop

Every training loop you will ever write — for XOR, for a CNN, for GPT — is the same four-beat heartbeat
you've known since P02, now one tidy line each: clear the old blame, predict and measure, trace the
blame back, take a step. Learn this little rhythm once and you can read any training script in the
world.

PyTorch's training loop is the same four beats from P06/P11, now one line each:
```
optimizer.zero_grad()      # clear old gradients (they accumulate otherwise!)
loss = criterion(pred, y)  # forward + loss
loss.backward()            # backprop (B2)
optimizer.step()           # update weights (Adam, P12)
```
We train the XOR net with Adam and watch the loss fall — compare to P11's hand-rolled loop.
""")

code(r'''
import torch, torch.nn as nn
torch.manual_seed(0)
class XORNet(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(2,8); self.fc2=nn.Linear(8,1)
    def forward(self,x): return self.fc2(torch.relu(self.fc1(x)))

net = XORNet()
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss()                       # sigmoid + BCE (P07) in one stable op
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)  # Adam from P12

for epoch in range(2001):
    optimizer.zero_grad()
    logits = net(X)
    loss = criterion(logits, Y)
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch:>4}  loss {loss.item():.4f}")

with torch.no_grad():
    preds = torch.sigmoid(net(X)).round().int().flatten().tolist()
print("\npredictions:", preds, " target [0,1,1,0] -> XOR solved with PyTorch in ~10 lines")
''')

readonly(r'''
# The SAME training in Keras (read-only):
model.compile(optimizer=keras.optimizers.Adam(0.05),
              loss=keras.losses.BinaryCrossentropy(from_logits=True))
model.fit(X, Y, epochs=2000, verbose=0)         # Keras hides the loop entirely
preds = (model.predict(X) > 0).astype(int)
# PyTorch = explicit loop you control;  Keras = .fit() convenience. Same math underneath.
''')

md(r"""
## B5 — GPU, eval mode, and saving (the practical bits)

The last few habits are the difference between a notebook experiment and a model that survives
contact with the real world: move work onto a GPU when one exists, flip the brain into "performance
mode" before showing it to users (so quirks like dropout switch off), and save what it learned so a
trained mind never has to start over. Small chores, but skipping them is how good models quietly go
wrong.

A few essentials you'll use constantly:
- **Device:** `tensor.to("cuda")` moves data to a GPU (this machine is CPU-only, so we just show the
  pattern with `torch.cuda.is_available()`).
- **Train vs eval mode:** `net.eval()` turns off dropout/uses running BatchNorm stats (P13);
  `net.train()` turns them back on. **Always** `eval()` for inference.
- **Save/load:** persist the learned weights with `state_dict`.
""")

code(r'''
import torch, torch.nn as nn, io
print("CUDA available here?", torch.cuda.is_available(), " (CPU-only box; pattern still applies)")
device = "cuda" if torch.cuda.is_available() else "cpu"

net = nn.Linear(2, 1)
net.eval()                                   # inference mode
print("module training mode:", net.training, " (False after .eval())")

# save & reload the weights (to an in-memory buffer here instead of disk)
buf = io.BytesIO(); torch.save(net.state_dict(), buf); buf.seek(0)
# weights_only=True is the safe default: it refuses to unpickle arbitrary code,
# loading only tensors/plain data. Always use it when loading checkpoints you
# didn't create yourself.
net2 = nn.Linear(2, 1); net2.load_state_dict(torch.load(buf, weights_only=True))
print("weights reloaded equal?", torch.allclose(net.weight, net2.weight))
''')

md(r"""
## Recap — from hand-built to production

| You hand-built (P10–P13) | PyTorch | Keras (read-only) |
|--------------------------|---------|-------------------|
| numpy arrays | `torch.tensor` | `tf.Tensor` |
| manual backprop | `loss.backward()` | `GradientTape` / `.fit()` |
| `W x + b` | `nn.Linear` | `keras.layers.Dense` |
| hand GD/Adam loop | `torch.optim.Adam` + loop | `model.compile/.fit` |
| dropout/eval toggle | `net.train()/.eval()` | automatic in `.fit/.predict` |

PyTorch gives you an explicit loop (what TinyGPT uses); Keras hides it. Same math — the math you now
understand from first principles.

## Common mistakes
1. **Forgetting `optimizer.zero_grad()`** → gradients accumulate across steps and training goes haywire.
2. **Calling `.backward()` on a non-scalar.** Reduce to a single number (e.g. `loss.mean()`) first.
3. **Leaving the model in train mode for inference** → dropout/BatchNorm corrupt predictions. Use `.eval()`.
4. **Doing math on tensors with `requires_grad=True` inside evaluation** without `torch.no_grad()` → wasted memory/graph.
5. **Mixing devices** (CPU tensor + GPU model) → runtime error. Keep data and model on the same device.

## Exercises (do them in new code cells)
1. Rebuild P11's 2-hidden-layer net in PyTorch and confirm it solves XOR.
2. Swap `Adam` for `torch.optim.SGD(lr=0.5)`. How many epochs to solve XOR now? (Recall P12.)
3. Verify autograd on `f(x)=x³` at `x=2`: build `x` with `requires_grad`, call `.backward()`, check `x.grad==12`.
4. Add `nn.Dropout(0.5)` to the XOR net. Compare predictions with `.train()` vs `.eval()`.
5. In the read-only Keras cell, identify which line corresponds to `optimizer.step()` in PyTorch.

---

**The mind is grown, trainable, and now holds professional tools** — it understands its own
machinery from first principles *and* can wield the libraries that build real models. So far it
thinks in plain numbers. Time to give it senses. Next we give it eyes:
[P15 — Convolutional Neural Networks](P15_CNNs.ipynb) — the architecture that sees images, with
convolution worked out by hand. (Then P16 gives it memory with RNNs, which leads the mind toward
attention and its true form, the transformer.)
""")

build("P14_PyTorch_Deep_Dive", execute=False)
