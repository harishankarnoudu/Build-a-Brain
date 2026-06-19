"""Builder for playbook/P15_CNNs.ipynb

The architecture that sees images. Convolution worked out BY HAND on a tiny 2D
grid (a real edge-detector kernel), then pooling, then the full conv->pool->FC
idea, with a PyTorch Conv2d that reproduces the hand convolution. Keras mirror
read-only. Keeps it conceptual since we're CPU-only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P15 — Convolutional Neural Networks  *(we give the mind eyes)*

> **The story so far.** In Part 2 we wired a brain: neurons (P10), the flash of learning that is
> backprop (P11), optimizers that learn faster (P12), and the tricks that keep a deep net standing
> (P13). The mind can now learn anything we can flatten into a row of numbers. But hand it a *photo*
> and it stumbles — a fully-connected net (P10) sees an image as one long anonymous list of pixels,
> blind to the fact that neighbouring pixels belong together and that a cat is a cat whether it sits
> top-left or bottom-right. So in this chapter we **give the mind eyes**: first it learns to feel
> **edges**, then it stitches edges into **shapes**, then shapes into **things**.

Picture how *you* recognise a cat in a photo. You don't reason about pixel #4,051 in isolation; your
eye sweeps the image looking for the same little features — a whisker here, an ear-edge there —
*wherever* they appear. **Convolutional Neural Networks (CNNs)** give the mind exactly that habit:
slide one small **filter** across the whole image, firing whenever its local pattern (an edge, a
texture) shows up, anywhere. That single idea is what powers image classification, object detection,
and medical-scan reading.

We work the core operation — **convolution** — by hand on a tiny grid using a real edge-detection
filter, add **pooling**, then watch a PyTorch `Conv2d` reproduce our by-hand numbers exactly. With
eyes done, P16 gives the mind **memory** — and shows us why memory alone wasn't enough.
""")

md(r"""
## B1 — Convolution by hand: slide a filter, take dot products

Run your fingertip down the boundary between a dark wall and a bright window — you *feel* the edge
the instant the brightness jumps. A CNN feels edges the same way, with a tiny detector dragged
across the picture. That detector is a **filter** (or kernel): a small grid of weights. **Convolution**
slides it over every position of the input, and at each spot computes the **dot product** (P01 — the
mind's one primitive sense, "how much do these two things agree?") of the filter with the patch it
covers. The result is a **feature map** that lights up exactly where the filter's pattern occurs.

**Worked example by hand.** Input `4×4`, a vertical-edge filter `3×3`:
```
input I =                 filter K (vertical edge) =
[ 0  0  1  1 ]            [ 1  0  -1 ]
[ 0  0  1  1 ]            [ 1  0  -1 ]
[ 0  0  1  1 ]            [ 1  0  -1 ]
[ 0  0  1  1 ]
```
Place K over the top-left `3×3` patch (rows 0-2, cols 0-2):
```
patch = [0 0 1; 0 0 1; 0 0 1]
dot = 1·0+0·0+(−1)·1  + 1·0+0·0+(−1)·1 + 1·0+0·0+(−1)·1 = −1−1−1 = −3
```
Slide one column right (cols 1-3): patch `[0 1 1; 0 1 1; 0 1 1]` →
`1·0−1·1 + 1·0−1·1 + 1·0−1·1 = −3`. The output `2×2` map will read `[[-3,-3],[-3,-3]]` — the filter
fires (non-zero) exactly along the vertical edge between the 0-region and the 1-region.
""")

code(r'''
import numpy as np
I = np.array([[0,0,1,1],
              [0,0,1,1],
              [0,0,1,1],
              [0,0,1,1]], float)
K = np.array([[1,0,-1],
              [1,0,-1],
              [1,0,-1]], float)         # detects vertical edges

def conv2d(I, K):
    kh, kw = K.shape; oh, ow = I.shape[0]-kh+1, I.shape[1]-kw+1
    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = (I[i:i+kh, j:j+kw] * K).sum()    # dot product of filter with patch
    return out

print("feature map:\n", conv2d(I, K))
print("hand top-left value = -3 (the filter responds to the vertical edge)")
''')

md(r"""
## B2 — Why CNNs win on images: weight sharing & locality

A toddler who learns what a *whisker* looks like doesn't need to relearn it for the left side of the
face and again for the right — one lesson, reused everywhere. That instinct is the whole reason
convolution beats brute force. Two ideas make it powerful:
- **Local connectivity** — each output looks at a small patch, matching how visual features are
  local (an edge spans a few pixels, not the whole image).
- **Weight sharing** — the *same* filter is used at every position, so the network needs far fewer
  parameters than a fully-connected layer, and a pattern learned in one place is detected
  everywhere (**translation invariance** — a cat is a cat top-left or bottom-right).

**By the numbers:** a fully-connected layer on a `100×100` image to 100 hidden units needs
`100·100·100 = 1,000,000` weights. A conv layer with ten `3×3` filters needs `10·3·3 = 90` weights —
and detects its patterns across the entire image. That efficiency is why CNNs dominated vision.
""")

code(r'''
fc_params   = 100*100*100
conv_params = 10*3*3
print("fully-connected layer params :", f"{fc_params:,}")
print("conv layer (10 3x3 filters)  :", f"{conv_params:,}")
print("reduction factor             :", f"{fc_params//conv_params:,}x fewer parameters")
''')

md(r"""
## B3 — Pooling: shrink the map, keep the strongest signal

Squint at a photo. Detail blurs away, but the *important* stuff — where the bright bits and strong
edges are — survives. The mind does this on purpose with **pooling**. **Max pooling** slides a window
(usually `2×2`) and keeps only the maximum in each — halving the resolution, cutting computation, and
buying a little tolerance to *where exactly* a feature sat. It has no learnable weights; it just
summarises "the strongest thing around here was…".

**Worked example by hand** on a `4×4`, `2×2` max-pool (stride 2):
```
[ 1  3 | 2  4 ]                 top-left  window max(1,3,5,6) = 6
[ 5  6 | 1  0 ]      ->         top-right window max(2,4,1,0) = 4
[ 2  1 | 0  8 ]                 bot-left  window max(2,1,7,3) = 7
[ 7  3 | 9  2 ]                 bot-right window max(0,8,9,2) = 9
```
Result `[[6,4],[7,9]]`.
""")

code(r'''
import numpy as np
X = np.array([[1,3,2,4],
              [5,6,1,0],
              [2,1,0,8],
              [7,3,9,2]], float)
def maxpool2x2(X):
    out = np.zeros((X.shape[0]//2, X.shape[1]//2))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = X[2*i:2*i+2, 2*j:2*j+2].max()
    return out
print("2x2 max pool:\n", maxpool2x2(X), "  (hand [[6,4],[7,9]])")
''')

md(r"""
## B4 — The CNN architecture, and PyTorch reproducing the hand convolution

Now the full life of a glance: **edges → shapes → things**. A typical CNN stacks
`[Conv → ReLU → Pool] × N → flatten → fully-connected → output`. The earliest conv layers feel raw
edges; the next layers fold edges into corners and textures; deeper still, those become object
*parts* (an ear, an eye); the top combines parts into whole objects and names them. The mind builds
a cat from the bottom up. To make sure we haven't been fooling ourselves, we now check that PyTorch's
`Conv2d` computes the *exact* convolution we did by hand in B1 — same filter, same numbers.
""")

code(r'''
import torch, torch.nn as nn, numpy as np
I = torch.tensor([[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,1,1]], dtype=torch.float32)
K = torch.tensor([[1,0,-1],[1,0,-1],[1,0,-1]], dtype=torch.float32)

conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
with torch.no_grad():
    conv.weight.copy_(K.view(1,1,3,3))           # set the filter to our hand kernel
out = conv(I.view(1,1,4,4))                       # (batch, channel, H, W)
print("PyTorch Conv2d feature map:\n", out.detach().view(2,2).numpy())
print("matches the hand answer [[-3,-3],[-3,-3]] -> nn.Conv2d IS the sliding dot product.")
''')

readonly(r'''
# A small CNN classifier in PyTorch (the standard shape) — runs conceptually:
import torch.nn as nn
cnn = nn.Sequential(
    nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28x28 -> 14x14
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14x14 -> 7x7
    nn.Flatten(),
    nn.Linear(32*7*7, 10),                                        # 10 classes (e.g. MNIST digits)
)
# Conv layers learn the FILTERS (we hand-set one above); training discovers good ones.
''')

readonly(r'''
# The SAME CNN in Keras (read-only — TF is blocked in this environment):
from tensorflow import keras
model = keras.Sequential([
    keras.layers.Conv2D(16, 3, activation="relu", padding="same", input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation="softmax"),
])
# Note Keras puts channels LAST (28,28,1); PyTorch puts channels FIRST (1,28,28).
''')

md(r"""
## Recap — convolutional networks

| Piece | What it does | Hand example |
|-------|--------------|--------------|
| Filter/kernel | small learnable pattern detector | B1 vertical-edge `3×3` |
| Convolution | slide filter, dot product each patch | B1 → feature map |
| Weight sharing | same filter everywhere → few params | B2 (10000× fewer) |
| ReLU | nonlinearity (P10) | between conv & pool |
| Max pooling | downsample, keep strongest | B3 |
| Conv→Pool stack → FC | edges → parts → objects → class | B4 |

## Common mistakes
1. **Channel-order confusion.** PyTorch is `(N, C, H, W)`; Keras/TF is `(N, H, W, C)`. Mixing them is a classic bug.
2. **Forgetting padding.** Without padding the map shrinks each layer; use `padding=1` for `3×3` to keep size.
3. **Flattening before the conv layers have reduced spatial size** → huge FC layer. Pool first.
4. **Expecting CNNs to handle arbitrary-length sequences.** They're built for grid data; sequences want RNNs/transformers (P16+).
5. **Hand-setting filters in practice.** We did it to verify; real CNNs *learn* filters via backprop.

## Exercises (do them in new code cells)
1. Apply the *horizontal*-edge filter `[[1,1,1],[0,0,0],[-1,-1,-1]]` to B1's input. Where does it fire (and why mostly zero here)?
2. Convolve a `5×5` input with a `3×3` filter — what output size? Derive the formula `out = in − k + 1`.
3. Do `2×2` average pooling (mean instead of max) on B3's grid by hand, then in code.
4. How many parameters in `nn.Conv2d(3, 8, kernel_size=5)` (with bias)? (Hint: `out·in·k·k + out`.)
5. Why does weight sharing give translation invariance? Explain with the cat-position example.

---

**The mind can see.** It feels edges, folds them into shapes, and names the things they make — and
it does this anywhere in the frame, because the same filter sweeps the whole image. But sight lives
in the *now*; a picture has no before or after. To read a sentence or follow a melody, the mind needs
to remember what just happened. Next, in
[P16 — Recurrent Networks (RNN/LSTM/GRU)](P16_RNNs.ipynb), **we give the mind memory** — and then
discover, painfully, why a single thread of memory wasn't enough. That ache is exactly what
**attention** was born to cure, the doorway into Part 3.
""")

build("P15_CNNs", execute=False)
