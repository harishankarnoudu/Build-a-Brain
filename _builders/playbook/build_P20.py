"""Builder for playbook/P20_Transformer_Bridge.ipynb

The hand-off. A short bridge that (1) recaps how every playbook piece maps onto a
transformer, (2) builds a complete tiny transformer block in numpy tying P13
(LayerNorm/residual) + P19 (self-attention) together, and (3) routes the learner
into the existing TinyGPT notebooks 00-13 with a clear map of which prior notebook
prepared each one. After P20 the learner does notebooks/01-13, then returns for P21+.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, reset, build

reset()

md(r"""
# P20 — From Attention to the Transformer  *(the assembly scene — every piece clicks together)*

> **The story so far.** We've raised this mind from a single bare number all the way to one that can
> read with attention. Look back at what it's grown: numbers in motion and the dot product (Part 0),
> first instincts and the line that explains the world (Part 1), a brain that learns by *predict →
> measure → step downhill*, with eyes and memory (Part 2), and words-as-places, meaning-as-direction,
> and the glance of attention (Part 3). Every organ is grown. This is **the assembly scene** — the
> moment in the story where the camera pulls back and every piece we forged separately clicks
> together into one shape. That shape has a name: the **transformer**.

This is a hinge notebook — short, and almost entirely recognition. Behind you: every ingredient of a
transformer, built and verified by hand. Ahead: the existing **TinyGPT notebooks (`notebooks/00`–`13`)**,
which assemble those ingredients into a real, trainable GPT. The thrill here is that you build *nothing
new* — you only watch the parts you already own snap into place.

This bridge does three things: (1) shows a **transformer block is just pieces you already built**,
(2) assembles a complete tiny block in NumPy to prove it, and (3) hands you a map into the TinyGPT
track noting which playbook notebook prepared each step.
""")

md(r"""
## B1 — A transformer block = pieces you already own

Watch the parts slide together. Here is a full transformer block with each sub-part labelled by the
chapter of the mind's life that grew it. Read it like an end-of-film montage where every character
returns — there is **nothing new**, only assembly:
```
x ──► LayerNorm ───────────────► self-attention ──► + x  (residual)
            (P13)                    (P19, NB00 B9)      (P13 B5)
  ──► LayerNorm ──► feed-forward (Linear→ReLU→Linear) ──► + x  (residual)
        (P13)              (P10 layers, P05 ReLU)            (P13)
```
Stack `N` of these, put a token+position embedding (P18, NB00 B7/B8) in front and a linear "LM
head" + softmax (NB00 B6/B10) at the end, train with cross-entropy (P03) and Adam (P12) via backprop
(P11) — and you have a GPT. That sentence names the entire model, and you've done every part.
""")

md(r"""
## B2 — Assemble a complete tiny transformer block (NumPy, by hand)

Talk is cheap — let's actually screw the parts together and turn it on. Here's one full block running
on 3 tokens of dimension 4: self-attention (P19, the glance) wrapped in LayerNorm (P13) and residuals
(P13), then a feed-forward network (P10) wrapped the same way. Watch the shape stay `(3, 4)` from end
to end — a block *refines* the mind's token-thoughts without reshaping them, which is precisely why
you can stack block on block forever.
""")

code(r'''
import numpy as np
rng = np.random.default_rng(0)
def softmax(z): z=z-z.max(axis=-1,keepdims=True); e=np.exp(z); return e/e.sum(axis=-1,keepdims=True)
def layernorm(x, eps=1e-5):
    mu=x.mean(-1,keepdims=True); var=((x-mu)**2).mean(-1,keepdims=True)
    return (x-mu)/np.sqrt(var+eps)

T, d = 3, 4
x = rng.normal(size=(T, d))                       # 3 token vectors

# --- learned matrices (random here; training would set them) ---
Wq,Wk,Wv = [rng.normal(0,0.5,(d,d)) for _ in range(3)]
W1 = rng.normal(0,0.5,(d, 4*d)); W2 = rng.normal(0,0.5,(4*d, d))   # FFN expands then contracts

def self_attention(h):
    Q,K,V = h@Wq, h@Wk, h@Wv                       # P19 / NB00 B9
    scores = (Q @ K.T) / np.sqrt(d)
    return softmax(scores) @ V

def feed_forward(h):
    return np.maximum(0, h @ W1) @ W2              # Linear -> ReLU -> Linear (P10)

# --- the block: each sublayer is  x = x + sublayer(LayerNorm(x))  (P13 residual + norm) ---
print("input shape :", x.shape)
x = x + self_attention(layernorm(x))               # attention sublayer
print("after attention sublayer:", x.shape)
x = x + feed_forward(layernorm(x))                 # feed-forward sublayer
print("after feed-forward sublayer:", x.shape)
print("\nshape preserved (3,4) -> blocks stack. THIS is notebooks/06_Transformer_Block.")
''')

code(r'''
# stacking N blocks is just calling the block N times — that's the whole "deep" in deep transformer
import numpy as np
rng = np.random.default_rng(0)
def softmax(z): z=z-z.max(axis=-1,keepdims=True); e=np.exp(z); return e/e.sum(axis=-1,keepdims=True)
def layernorm(x,eps=1e-5):
    mu=x.mean(-1,keepdims=True); var=((x-mu)**2).mean(-1,keepdims=True); return (x-mu)/np.sqrt(var+eps)
T,d=3,4; x0=rng.normal(size=(T,d))
blocks=[ [rng.normal(0,0.3,(d,d)) for _ in range(3)] + [rng.normal(0,0.3,(d,4*d)),rng.normal(0,0.3,(4*d,d))] for _ in range(4)]
def block(h,P):
    Wq,Wk,Wv,W1,W2=P
    Q,K,V=h@Wq,h@Wk,h@Wv; a=softmax((Q@K.T)/np.sqrt(d))@V; h=h+a
    h=h+np.maximum(0,layernorm(h)@W1)@W2; return h
h=x0
for i,P in enumerate(blocks):
    h=block(layernorm(h),P)
print("passed through", len(blocks), "stacked blocks; final shape", h.shape, "(still 3x4)")
print("a 4-block transformer in a dozen lines — every op from the playbook.")
''')

md(r"""
## B3 — Your route through the TinyGPT notebooks

The mind is assembled — now go give it life. Work through `notebooks/00`–`13` in order; this is the
hand-off, where the playbook passes the baton to the TinyGPT track. The map below shows which chapter
of the mind's upbringing prepared each step, so nothing will look like a stranger:

| TinyGPT notebook | Builds | Prepared by |
|------------------|--------|-------------|
| `00_Introduction` | the 5 core ops, end-to-end | all of Part 0, P10–P12 |
| `01_Tokenizer` | text → token ids | P17 |
| `02_Embeddings` | token → vector | P18 |
| `03_Neural_Network_Basics` | a mini LM, no attention | P10, P11 |
| `04_Self_Attention` | Q/K/V attention from scratch | P19, this notebook |
| `05_MultiHead_Attention` | parallel attention heads | P19 + P01 (reshape) |
| `06_Transformer_Block` | attention + FFN + residual + LN | **B2 above**, P13 |
| `07_Positional_Encoding` | inject token order | NB00 B8 |
| `08_Build_TinyGPT` | assemble the full model | this whole bridge |
| `09_Train_TinyGPT` | training loop on real text | P06, P11, P12 |
| `10_Text_Generation` | sampling strategies | P03, → P26 |
| `11_Model_Interpretability` | attention maps, embeddings | P18, P19 |
| `12_SecurityGPT` | domain fine-tuning | P05, → P22 |
| `13_Modern_LLM_FineTuning` | LoRA/RLHF/RAG overview | → P21–P28 |

**When you finish `notebooks/13`, return here for Part 5** (P21–P27): pretraining & scaling laws,
supervised fine-tuning, LoRA/QLoRA, RLHF (reward models, PPO, DPO), RAG & agents, prompt
engineering & decoding, and evaluation/safety — then Part 6 (P28–P29) on serving & MLOps.
""")

md(r"""
## Recap — the bridge

- A transformer block = **self-attention (P19) + feed-forward (P10), each wrapped in LayerNorm +
  residual (P13)**. You assembled a complete one in NumPy (B2) and stacked four (B2 cont.).
- A GPT = embeddings (P18) + positional encoding + `N` blocks + LM head + softmax, trained with
  cross-entropy (P03) and Adam (P12) via backprop (P11).
- **There is no missing piece.** Everything the TinyGPT notebooks use, you built and verified here.

## Common mistakes
1. **Skipping the residual/LayerNorm wrapping** — without it deep transformers don't train (P13).
2. **Thinking multi-head attention is new math** — it's the same attention run in parallel on slices of the vector (P05 reshape + P19).
3. **Forgetting positional info** — self-attention alone is order-blind (NB00 B8).
4. **Expecting one block to be "deep".** Depth comes from stacking many identical blocks (B2 cont.).
5. **Treating the LM head as special** — it's one linear layer (P01) mapping to vocab size, then softmax.

## Exercises (do them in new code cells)
1. In B2, add a second feed-forward sublayer. Does the shape still come out `(3,4)`?
2. Print the attention weights inside B2's `self_attention`. Which token attends to which?
3. Change `T` to 5 tokens in B2. What changes, and what stays the same?
4. Open `notebooks/04_Self_Attention` and confirm its Q/K/V code matches your B2 `self_attention`.
5. Sketch (in a markdown cell) the full GPT data flow from token ids to next-token probabilities, naming the playbook notebook for each arrow.

---

**The pieces have clicked together — the mind has its true form.** Everything you raised across
Parts 0–3 now stands as one architecture, and there is no missing organ. **Go build TinyGPT
(`notebooks/00`–`13`) and watch it wake up** — every line of that model is something you understand
from first principles, because you grew each part by hand. When the mind is alive and generating
text, **come back here for Part 5**, where the real adventure begins: we **feed it the whole
internet** — [P21 — Pretraining & Scaling](P21_Pretraining_and_Scaling.ipynb) — the same training
loop you already know, run a billion-fold bigger.
""")

build("P20_Transformer_Bridge", execute=False)
