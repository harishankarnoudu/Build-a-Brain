"""Builder for playbook/P19_Seq2Seq_and_Attention.ipynb

How translation models exposed the fixed-vector bottleneck, and how attention
solved it. We build the encoder-decoder idea, show the bottleneck, then implement
attention BY HAND (weighted sum of encoder states by softmaxed similarity) - the
exact mechanism of Notebook 00 B9, here motivated from the translation problem.
Direct on-ramp to P20 and notebooks/04.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, reset, build

reset()

md(r"""
# P19 — Seq2Seq & the Birth of Attention  *(the mind learns to pay attention)*

> **The story so far.** In P18 words became *places* and meaning became a *direction*. But each word
> still had only one fixed home, and the mind's old memory (P16) still faded across a long sentence.
> Now we set it the hardest reading task yet — **translation** — and that task breaks its fading
> memory wide open. Out of the wreckage, the mind discovers the move it has been missing since P16:
> instead of cramming everything it read into one shrinking note, it learns to **look back over the
> whole sentence and decide, word by word, what matters right now.** This is the moment **the mind
> learns to pay attention** — and it changes everything that follows.

Picture a translator working from English to French. A weak translator reads the whole English
sentence, memorises a single fuzzy impression, then tries to write French from that alone. A great
translator keeps the source in view and, for each French word, *glances back* at the relevant English
words. The first is **sequence-to-sequence (seq2seq)**: an **encoder** RNN reads the input into one
summary vector, a **decoder** RNN writes the output from it. It worked — until sentences grew long
and that single summary couldn't hold them. **Attention** is the second translator: let the decoder
look back at *all* encoder states, weighted by relevance. That move is the beating heart of the
transformer.

We build attention *out of the translation problem itself*, so by the time you reach Notebook 00 B9 /
`notebooks/04` you'll know not just *how* it works but *why it had to be born*.
""")

md(r"""
## B1 — Encoder–decoder: compress, then generate

Here's the weak translator, made precise. Read the whole sentence, squeeze it into one impression,
then speak from that impression alone. The seq2seq recipe (building on the RNN of P16):
1. **Encoder** reads the input sequence one token at a time, updating a hidden state. Its **final**
   hidden state is a fixed-size summary `c` ("context vector") of the whole input.
2. **Decoder** starts from `c` and generates the output sequence one token at a time.

```
"the cat sat"  ──encoder──►  c (one vector)  ──decoder──►  "le chat assis"
```
Everything the decoder knows about the input must squeeze through that single `c`.
""")

code(r'''
import numpy as np
rng = np.random.default_rng(0)
# toy encoder: read 3 input tokens, keep a running hidden state (RNN from P16)
def relu(z): return np.maximum(0,z)
Wx = rng.normal(0,0.5,(4,4)); Wh = rng.normal(0,0.5,(4,4))
inputs = rng.normal(size=(3,4))          # 3 input-token vectors
h = np.zeros(4); states = []
for x in inputs:
    h = np.tanh(Wx@x + Wh@h); states.append(h.copy())
context = states[-1]                      # the single summary vector c
print("encoder produced", len(states), "hidden states, one per input token")
print("context vector c (final state only):", np.round(context,3))
print("-> the decoder will see ONLY this one vector. That's the bottleneck.")
''')

md(r"""
## B2 — The bottleneck: one vector can't hold a long sentence

Try to repeat a 40-word sentence back after hearing it once — the front of it has already slipped
away by the time you reach the end. The mind has the same problem: cramming a long sentence into one
fixed vector loses information, especially about early words (the very vanishing-memory ache of P16).
As sentences grew, translation quality collapsed. The diagnosis writes the cure: the decoder
shouldn't lean on a *single* summary — it should be able to **look back at every encoder state** and
focus on the relevant ones for each word it writes.
""")

code(r'''
import numpy as np
# illustrate: reconstruction error grows as we force longer sequences through one fixed vector
rng = np.random.default_rng(1)
print("how much input information survives one fixed-size (dim-16) summary?")
for seq_len in [3, 10, 30, 60]:
    info = rng.normal(size=(seq_len, 16))
    summary = info.mean(0)                       # crude 'compression' into one vector
    # try to recover each token from the summary alone -> error rises with length
    recon_err = np.mean([np.linalg.norm(summary - tok) for tok in info])
    print(f"  seq length {seq_len:>2}: avg reconstruction error = {recon_err:.2f}")
print("longer input -> worse single-vector summary. Attention removes this ceiling.")
''')

md(r"""
## B3 — Attention, derived from the fix

This is the move itself — the great translator's glance, written as math. When the decoder produces
output word `t`, instead of leaning on the one stale summary `c`, it builds a **custom context
vector** fresh for this word: a weighted blend of *all* encoder states, each weighted by how relevant
it is to what the decoder needs *right now*. The mind asks "which words should I look at for this?"
and answers with the dot product (P01, its sense of agreement). The three steps (identical to
Notebook 00 B9):
```
1. scores[i]  = decoder_query · encoder_state[i]     (relevance via dot product, P01)
2. weights    = softmax(scores)                       (turn into weights that sum to 1, P03/NB00)
3. context_t  = Σ weights[i] · encoder_state[i]       (weighted blend of ALL encoder states)
```
No information bottleneck (every state is reachable) and no distance decay (any state can get high
weight). We compute one full attention step by hand.

**Worked example by hand.** Decoder query `q=[1,0]`, three encoder states:
```
e0=[1,0]  e1=[0,1]  e2=[1,1]
scores = q·e = [1, 0, 1]
softmax([1,0,1]) = [0.422, 0.155, 0.422]   (e0 and e2 most relevant)
context = 0.422·[1,0] + 0.155·[0,1] + 0.422·[1,1] = [0.845, 0.578]
```
""")

code(r'''
import numpy as np
def softmax(z): z=z-z.max(); e=np.exp(z); return e/e.sum()

q = np.array([1.0, 0.0])                       # what the decoder is looking for now
E = np.array([[1.0,0.0],[0.0,1.0],[1.0,1.0]])  # encoder states (one per input token)

scores  = E @ q                                # step 1: relevance of each encoder state
weights = softmax(scores)                      # step 2: attention weights
context = weights @ E                          # step 3: weighted blend

print("scores          :", scores, " (hand [1,0,1])")
print("attention weights:", np.round(weights,3), " (hand [0.422,0.155,0.422], sum",round(weights.sum(),3),")")
print("context vector   :", np.round(context,3), " (hand [0.845,0.578])")
print("\n-> the decoder now has a CUSTOM summary focused on the relevant input words. No bottleneck.")
''')

md(r"""
## B4 — From attention to self-attention to the transformer

Once the mind can glance at *another* sentence, an audacious thought follows: why not let a sentence
glance at **itself**? Let every word look at every other word in the *same* sequence, asking "which of
my neighbours change what I mean?" — so "bank" can finally read "river" two words over and settle into
the right meaning. That's **self-attention** (Notebook 00 B9). And if this glance is so powerful, why
keep dragging along the slow, sequential RNN at all? The 2017 paper's answer — *"Attention Is All You
Need"* — was to throw recurrence out entirely:
That's **self-attention** (Notebook 00 B9). And if attention is so powerful, why keep the slow,
sequential RNN at all? The 2017 paper's answer — *"Attention Is All You Need"* — was to drop
recurrence entirely:
- **Queries, Keys, Values** are all computed from the input by learned matrices (Notebook 00 B9).
- Stacking self-attention + feed-forward layers with residuals & LayerNorm (P13) = a **transformer
  block** (`notebooks/06`).
- No recurrence → the whole sequence is processed **in parallel** → trains fast on huge data → GPT.

You now have every prerequisite for the transformer. The next notebook (P20) makes the hand-off.
""")

code(r'''
import numpy as np
# self-attention = attention where queries, keys, values all come from the SAME sequence
def softmax(z): z=z-z.max(axis=-1,keepdims=True); e=np.exp(z); return e/e.sum(axis=-1,keepdims=True)
X = np.array([[1.0,0.0],[0.0,1.0],[1.0,1.0]])    # 3 tokens attend to each other
scores = (X @ X.T) / np.sqrt(X.shape[1])          # all pairwise relevances, scaled (NB00 B9)
W = softmax(scores)
out = W @ X
print("self-attention weights (each row = a token deciding where to look):\n", np.round(W,3))
print("output (each token = blend of all tokens):\n", np.round(out,3))
print("\nthis is EXACTLY Notebook 00 B9 — you derived it from the translation bottleneck.")
''')

md(r"""
## Recap — the road to transformers

| Stage | Mechanism | Problem / fix |
|-------|-----------|---------------|
| Seq2seq | encoder→`c`→decoder | works for short sequences |
| Bottleneck | one fixed vector | loses long-sentence info |
| Attention | decoder blends ALL encoder states | no bottleneck, no decay |
| Self-attention | sequence attends to itself | context-aware tokens |
| Transformer | self-attention + FFN + residual/LN, no RNN | parallel, scalable → GPT |

## Common mistakes
1. **Thinking attention replaced *all* of seq2seq at once.** It first *augmented* RNN seq2seq; transformers later dropped the RNN.
2. **Confusing the encoder's final state with attention.** Attention uses *every* encoder state, not just the last.
3. **Forgetting the softmax.** Raw scores aren't weights; softmax makes them sum to 1 (P03).
4. **Missing why parallelism matters.** No recurrence = process all tokens at once = train on far more data (the real unlock).
5. **Overlooking scaling by √d.** Without it, large dot products make softmax too peaked (Notebook 00 B9).

## Exercises (do them in new code cells)
1. Redo B3 with query `q=[0,1]`. Which encoder state now gets the most weight? Why?
2. In B3, what happens to the weights if all encoder states are identical? Compute by hand.
3. Add scaling `/√2` to B3's scores before softmax. How do the weights change?
4. In B4's self-attention, change token 2 to `[5,5]`. Recompute who it attends to most.
5. Explain in one sentence why a transformer trains faster than an RNN on a million sentences.

---

**The mind has learned to pay attention** — to look back over everything it has read and decide, for
each new word, exactly what matters. With that one move, its fading memory is healed and its slow
sequential reading is set free. Every separate gift we've raised in it — numbers (Part 0), instincts
(Part 1), a brain with eyes and memory (Part 2), and now meaning and attention (Part 3) — is finally
ready to snap together into one shape. Next, in
[P20 — From Attention to the Transformer (the bridge)](P20_Transformer_Bridge.ipynb), comes **the
assembly scene**: every piece clicks into place, and we hand the mind off to your **TinyGPT notebooks
(00–13)**, where it is built, trained, and brought fully to life.
""")

build("P19_Seq2Seq_and_Attention", execute=False)
