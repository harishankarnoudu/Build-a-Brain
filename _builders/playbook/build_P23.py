"""Builder for playbook/P23_LoRA_and_Quantization.ipynb

Parameter-efficient fine-tuning. LoRA's low-rank update W + (alpha/r)*B@A worked
out by hand with real parameter-count savings, a numerical demo that a low-rank
B@A can carry a weight change, then quantization (float -> int8 with scale &
zero-point, by hand) and the QLoRA idea. HF/peft code read-only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P23 — PEFT: LoRA / QLoRA / Quantization  *(reshape a giant mind on a laptop — add footpaths, don't move the mountain)*

> **The story so far.** In P22 the mind learned manners — but it did so as a *giant*. To send it back
> to school the ordinary way, you'd have to retrain every one of its billions of weights at once. The
> mind is now a mountain, and full fine-tuning means re-sculpting the whole mountain — you need a
> supercomputer just to hold the dust. This chapter asks a cheeky question: what if you don't move the
> mountain at all? What if you just lay down a few **footpaths** across it?

Imagine you want to change how people traverse a mountain. You *could* bulldoze and re-grade the
entire slope (full fine-tuning, P22) — for a 7-billion-parameter model that's 28 GB of gradients and
optimizer state, far beyond one consumer GPU. Or you could leave the mountain frozen exactly as it is
and carve a few **small new trails** that redirect the traffic. That's the whole idea of
**Parameter-Efficient Fine-Tuning (PEFT)**: freeze the giant pretrained mind and train only a *tiny*
number of new parameters. The dominant method is **LoRA** (Low-Rank Adaptation) — the footpaths.
Combined with **quantization** (storing the frozen mountain in coarser 4-bit numbers) you get
**QLoRA** — fine-tuning a 7B model on a single 24 GB GPU, the kind under your desk.

We work the LoRA math by hand (the parameter savings are dramatic and exact), prove numerically that
a low-rank update can carry a real weight change, then do **int8 quantization by hand**. This is the
technical core of `notebooks/13`.
""")

md(r"""
## B1 — The LoRA idea: a big weight update is secretly low-rank

The surprising thing about the footpaths is how *little* dirt you actually have to move. When a mind
adapts to a new task, the change it needs — the difference between "before" and "after" — turns out
to be far simpler than the mind itself. Fine-tuning changes a weight matrix `W` by some update `ΔW`:
`W_new = W + ΔW`. The LoRA insight: that update `ΔW` doesn't need full rank — it can be
well-approximated by the product of two **skinny** matrices:
```
ΔW ≈ B · A          where  W is (d×k),  B is (d×r),  A is (r×k),  and r ≪ d, k
W_new = W + (α/r)·B·A      (α is a scaling constant; W stays FROZEN, only A and B are trained)
```
Instead of training `d×k` numbers, you train only `d×r + r×k`. With `r` tiny (4, 8, 16), that's a
massive reduction.

**Worked example by hand.** A `1000×1000` weight (`d=k=1000`) with rank `r=8`:
```
full ΔW    : 1000 × 1000               = 1,000,000 trainable numbers
LoRA B·A   : 1000×8  +  8×1000         = 8,000 + 8,000 = 16,000 numbers
reduction  : 1,000,000 / 16,000        ≈ 62× fewer parameters to train
```
""")

code(r'''
d = k = 1000; r = 8
full = d*k
lora = d*r + r*k
print(f"full fine-tune of this layer : {full:,} trainable params")
print(f"LoRA (rank {r})              : {lora:,} trainable params")
print(f"reduction                    : {full/lora:.0f}x fewer")
print("\nacross a whole 7B model, LoRA typically trains <1% of the parameters.")
''')

md(r"""
## B2 — LoRA in action: a rank-1 update carries a real change (by hand + verified)

A single ridge-line trail, drawn with almost no dirt, can still redirect a whole mountainside of
hikers. Why can a *low-rank* `B·A` capture a useful update? Because many weight changes during
fine-tuning really are low-rank — the mind's "what's new for this task" lives in a handful of
directions, not all of them. We demonstrate the mechanism with the cleanest case, **rank 1**.

**Worked example by hand.** Let `B = [[1],[2]]` (2×1) and `A = [[3, 4]]` (1×2). Then
```
ΔW = B·A = [[1],[2]] · [[3,4]] = [[1·3, 1·4], [2·3, 2·4]] = [[3, 4], [6, 8]]
```
A full `2×2` change (4 numbers) was produced from just `2 + 2 = 4` LoRA numbers here — and for big
matrices with small `r` the saving is enormous. The frozen `W` plus this `ΔW` gives the adapted
layer.
""")

code(r'''
import numpy as np
B = np.array([[1.0],[2.0]])       # (2,1)
A = np.array([[3.0, 4.0]])        # (1,2)
dW = B @ A                        # rank-1 update
print("B @ A =\n", dW, "  (hand [[3,4],[6,8]])")
print("rank of this update:", np.linalg.matrix_rank(dW), " (rank 1 — built from skinny B,A)")

# frozen W plus the LoRA update = adapted weight
W = np.array([[10.,0.],[0.,10.]])
alpha, rr = 2.0, 1
W_new = W + (alpha/rr)*dW
print("\nW (frozen):\n", W)
print("W + (alpha/r)·B@A (adapted):\n", W_new)
print("only A and B were trained; W never moved -> tiny memory footprint.")
''')

code(r'''
# prove a low-rank adapter can FIT a target change: learn B,A so that B@A approximates a target ΔW
import numpy as np
rng = np.random.default_rng(0)
d, r = 6, 2
target = rng.normal(size=(d, d)); target = target @ target.T      # a (low-ish rank-friendly) target
B = rng.normal(0, 0.1, (d, r)); A = rng.normal(0, 0.1, (r, d)); lr = 0.01
for step in range(2000):
    pred = B @ A
    err = pred - target
    B -= lr * err @ A.T            # gradients of ||B@A - target||^2
    A -= lr * B.T @ err
print("after training the rank-2 adapter to match a target update:")
print("  relative error ||B@A - target|| / ||target|| =",
      round(np.linalg.norm(B@A - target)/np.linalg.norm(target), 3))
print("  (small -> two skinny matrices captured most of a full d×d change)")
''')

md(r"""
## B3 — Quantization: store weights in 8 (or 4) bits instead of 32

Footpaths shrink how much you *re-train*. Quantization shrinks how much the frozen mountain *weighs*
in memory. Think of it like saving a photo as a smaller file: you keep a coarser palette of colours,
the picture looks almost identical, but it takes a quarter of the disk. Weights are usually 32-bit
floats. **Quantization** stores them in low precision (int8, int4), cutting memory ~4–8×. The mapping
for int8: find the range, pick a **scale** so the max value maps to 127, round to integers, and store
the scale to convert back.
```
scale = max(|W|) / 127
W_int8 = round(W / scale)          # integers in [-127, 127]
W_dequant = W_int8 · scale          # approximate reconstruction
```
**Worked example by hand.** `W = [0.5, -1.0, 0.25]`, `max|W| = 1.0`:
```
scale = 1.0 / 127 = 0.007874
W/scale = [63.5, -127, 31.75]  -> round -> [64, -127, 32]
dequant = [64, -127, 32]·0.007874 = [0.5039, -1.0, 0.2520]   (tiny rounding error)
```
""")

code(r'''
import numpy as np
W = np.array([0.5, -1.0, 0.25])
scale = np.abs(W).max() / 127
W_int8 = np.round(W / scale).astype(int)
W_dequant = W_int8 * scale
print("original   :", W)
print("scale      :", round(scale, 6), " (hand 0.007874)")
print("int8 codes :", W_int8, " (hand [64, -127, 32]) — these are what get stored")
print("dequantized:", np.round(W_dequant, 4), " (hand [0.5039, -1.0, 0.2520])")
print("max error  :", round(np.abs(W - W_dequant).max(), 4), " (small — 8 bits is usually plenty)")
print("\nmemory: float32 = 4 bytes/weight -> int8 = 1 byte/weight = 4x smaller (int4 = 8x).")
''')

md(r"""
## B4 — QLoRA: quantization + LoRA together

Now put the two tricks together — squeeze the mountain small enough to fit on your desk, *then* draw
the footpaths across it. **QLoRA** is the combination that put big-model fine-tuning on consumer
hardware:
1. **Quantize the frozen base model to 4-bit** — it now fits in a fraction of the memory.
2. **Add LoRA adapters in higher precision** and train *only* those.

The huge base never receives gradients (it's frozen and tiny in memory), while the small adapters do
the learning in full precision. Result: fine-tune a 7B–70B model on one GPU. This is the practical
recipe most people use today, exposed in `notebooks/13`.
""")

readonly(r'''
# QLoRA with HuggingFace peft + bitsandbytes (read-only — these libs are blocked here):
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")     # B3 idea, 4-bit
base = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-7B", quantization_config=bnb)

lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj","v_proj"])  # B1/B2: rank-8 adapters
model = get_peft_model(base, lora)
model.print_trainable_parameters()   # prints something like "trainable: 0.06% of all params"
# then train with SFTTrainer (P22) — only the LoRA matrices A,B update.
''')

md(r"""
## Recap — efficient fine-tuning

| Technique | Idea | Win |
|-----------|------|-----|
| LoRA | update = skinny `B·A`, freeze `W` | train <1% of params |
| rank `r` | small (4–16) controls adapter size | tune capacity vs cost |
| Quantization | floats → int8/int4 + scale | 4–8× less memory |
| QLoRA | 4-bit frozen base + LoRA adapters | fine-tune 7B+ on one GPU |

## Common mistakes
1. **Setting LoRA rank too high** → you lose the efficiency; start small (8) and raise only if underfitting.
2. **Forgetting the `α/r` scaling** → adapter has the wrong magnitude; keep `alpha` and `r` consistent.
3. **Quantizing the LoRA adapters too** → they need precision to learn; only the *base* is 4-bit in QLoRA.
4. **Expecting to merge adapters across different base models.** Adapters are tied to the base they trained on.
5. **Assuming int8 always loses accuracy.** For inference it's usually negligible; for training, keep adapters high-precision.

## Exercises (do them in new code cells)
1. Compute LoRA params for a `4096×4096` layer at `r=16`, and the reduction factor, by hand; verify.
2. Build a rank-1 `ΔW` from `B=[[2],[0],[1]]`, `A=[[1,3]]`. What is its shape and rank?
3. Quantize `W=[2.0, -0.5, 1.5]` to int8 by hand (find the scale), then verify and report the max error.
4. In B2's adapter-fitting cell, set `r=1`. Does the relative error get worse? Why?
5. Explain in one sentence why QLoRA's frozen base can be 4-bit while the adapters stay 16-bit.

---

**The giant can now be reshaped cheaply — but reshaped toward *what*?** We've taught it manners and
made it affordable to tune; the one thing we still can't write down as a target is *taste*. Next:
[P24 — RLHF: Reward Models, PPO & DPO](P24_RLHF.ipynb) — teaching the mind what people actually
*prefer*, with the reward model and the DPO loss worked out by hand.
""")

build("P23_LoRA_and_Quantization", execute=False)
