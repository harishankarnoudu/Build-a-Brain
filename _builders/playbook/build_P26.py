"""Builder for playbook/P26_Prompt_and_Decoding.ipynb

How the model turns next-token probabilities into actual text, all by hand:
greedy, temperature, top-k, top-p (nucleus). Then prompt engineering (zero/few-shot,
chain-of-thought) shown to change outputs. Connects to notebooks/10_Text_Generation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, reset, build

reset()

md(r"""
# P26 — Prompt Engineering & Decoding  *(the dials that set its mood)*

> **The story so far.** The mind has manners, taste, a library, and hands (P22–P25). But you may have
> noticed two different-sounding minds can sit behind the *exact same weights*. One answers a factual
> question crisply and identically every time; another, asked to write a poem, surprises you. Nothing
> in the brain changed — only the **dials** on the outside. Every word the mind speaks starts as a
> *probability over what comes next*; how we turn that cloud of maybes into a single chosen word is a
> mood knob, and so is the way you *phrase* the request. This chapter is about steering the mind
> without changing a single weight.

The model outputs a probability for every next token (Notebook 00 B10). **Decoding** is how we turn
that distribution into actual generated text — and the strategy dramatically changes the result, from
robotic repetition to creative prose. It's a dial: crank it one way and the mind is a careful clerk,
crank it the other and it's a daydreamer. **Prompt engineering** is the complementary dial: changing
the *input* wording to steer the output without touching the weights.

We implement every decoding strategy — **greedy, temperature, top-k, top-p** — by hand on one tiny
distribution so you see exactly what each does, then show prompting patterns. This is the math behind
`notebooks/10_Text_Generation`.
""")

md(r"""
## B1 — Greedy decoding: always take the most likely token

This is the mind on its most cautious setting — the clerk who, at every fork, always takes the road
it's surest of. The simplest strategy: pick the **argmax** every step. Deterministic and safe, but
repetitive and dull — it can get stuck in loops ("the the the") and never surprises you.

**Worked example by hand.** Next-token probs over `["cat","sat","mat","ran"]` = `[0.5, 0.3, 0.15,
0.05]`. Greedy picks `"cat"` (the max) — every single time, no matter how many times you run it.
""")

code(r'''
import numpy as np
vocab = ["cat","sat","mat","ran"]
probs = np.array([0.5, 0.3, 0.15, 0.05])
print("greedy pick:", vocab[probs.argmax()], " (always 'cat' — deterministic)")
''')

md(r"""
## B2 — Temperature: a creativity dial

This is *the* mood knob. Turn it down and the mind is sober and focused; turn it up and it loosens,
takes chances, free-associates. Sampling adds randomness — but how much? **Temperature** `T` reshapes
the distribution *before* sampling by dividing the logits by `T`:
```
probs = softmax(logits / T)
```
- `T < 1` sharpens (more confident, closer to greedy).
- `T = 1` leaves it unchanged.
- `T > 1` flattens (more random, more creative/risky).

**Worked example by hand** with logits `[2, 1, 0]` (which softmax to `[0.665, 0.245, 0.090]` at T=1):
```
T=0.5: logits/0.5 = [4,2,0]  -> softmax = [0.867, 0.117, 0.016]   (sharper -> more 'cat')
T=2.0: logits/2.0 = [1,0.5,0]-> softmax = [0.506, 0.307, 0.186]   (flatter -> more variety)
```
""")

code(r'''
import numpy as np
def softmax(z): z=z-z.max(); e=np.exp(z); return e/e.sum()
logits = np.array([2.0, 1.0, 0.0])
for T in [0.5, 1.0, 2.0]:
    print(f"T={T}: {np.round(softmax(logits/T), 3)}")
print("hand: T=0.5 -> [0.867,0.117,0.016] (peaky);  T=2 -> [0.506,0.307,0.186] (flat)")
print("low T = focused/repetitive, high T = diverse/risky.")
''')

md(r"""
## B3 — Top-k sampling: only consider the k best tokens

A daydreaming mind sometimes blurts out something truly bizarre — that one-in-a-thousand word it never
should have considered. Top-k is a sensible chaperone: let the mind be spontaneous, but only among
the few options that actually make sense. Pure sampling can occasionally pick a terrible
low-probability token. **Top-k** clips to the `k` highest-probability tokens, renormalises, and
samples among those — keeping variety while banning the junk.

**Worked example by hand.** Probs `[0.5, 0.3, 0.15, 0.05]`, `k=2`. Keep the top 2 (`cat 0.5`, `sat
0.3`), drop the rest, renormalise:
```
kept = [0.5, 0.3] -> divide by 0.8 -> [0.625, 0.375]
```
Now we sample only between "cat" and "sat".
""")

code(r'''
import numpy as np
vocab = ["cat","sat","mat","ran"]; probs = np.array([0.5,0.3,0.15,0.05])
def top_k(probs, k):
    idx = np.argsort(probs)[::-1][:k]            # indices of the k largest
    kept = np.zeros_like(probs); kept[idx] = probs[idx]
    return kept / kept.sum()
p2 = top_k(probs, 2)
print("top-2 renormalised:", np.round(p2, 3), " (hand [0.625, 0.375, 0, 0])")
print("only 'cat'/'sat' can be sampled now; 'mat'/'ran' are banned.")
''')

md(r"""
## B4 — Top-p (nucleus) sampling: keep the smallest set covering p of the mass

A fixed chaperone is too rigid: sometimes only two words are sensible, sometimes twenty are. A wiser
rule adapts to the moment — keep just enough options to cover *most* of what the mind believes, no
more. Top-k's fixed `k` is clumsy — sometimes 2 tokens hold all the probability, sometimes you want 20.
**Top-p** (nucleus) instead keeps the *smallest set of tokens whose probabilities sum to ≥ p*, then
renormalises. It adapts to how peaked the distribution is — the modern default.

**Worked example by hand.** Probs `[0.5, 0.3, 0.15, 0.05]`, `p=0.9`. Add from the top until ≥ 0.9:
```
0.5            = 0.50   (< 0.9, keep going)
0.5+0.3        = 0.80   (< 0.9, keep going)
0.5+0.3+0.15   = 0.95   (≥ 0.9, stop) -> nucleus = {cat, sat, mat}
renormalise [0.5,0.3,0.15]/0.95 = [0.526, 0.316, 0.158]
```
""")

code(r'''
import numpy as np
vocab = ["cat","sat","mat","ran"]; probs = np.array([0.5,0.3,0.15,0.05])
def top_p(probs, p):
    order = np.argsort(probs)[::-1]
    cum = np.cumsum(probs[order])
    cutoff = np.searchsorted(cum, p) + 1          # how many tokens to keep
    keep = order[:cutoff]
    out = np.zeros_like(probs); out[keep] = probs[keep]
    return out / out.sum()
pp = top_p(probs, 0.9)
print("top-p=0.9 nucleus:", np.round(pp, 3), " (hand [0.526,0.316,0.158,0])")
print("kept the smallest set covering 90% of the mass — adapts to the distribution.")
''')

md(r"""
## B5 — Prompt engineering: steering output without retraining

The decoding dials set the mind's *mood*; the prompt sets its *brief*. Same trick a good manager uses
on a brilliant new hire — the talent is fixed, but how you frame the task changes everything you get
back. Same model, different *prompt*, very different results. The core techniques:
- **Zero-shot** — just ask. ("Classify the sentiment: 'I loved it.'")
- **Few-shot** — show examples first; the model imitates the pattern (in-context learning).
- **Chain-of-thought** — ask it to "think step by step", which dramatically improves reasoning by
  letting it generate intermediate steps before the answer.
- **Role / system prompts** — "You are a careful security analyst…" sets behaviour.

We illustrate the *structure* of these prompts (the actual LLM call is read-only elsewhere).
""")

code(r'''
zero_shot = "Classify sentiment (positive/negative): 'The food was cold.'\nAnswer:"

few_shot = """Classify sentiment (positive/negative):
'Loved every minute!' -> positive
'Terrible service.' -> negative
'The food was cold.' ->"""

cot = """Q: A shop had 12 apples, sold 5, then got 8 more. How many now?
Let's think step by step:
1) start 12
2) sold 5 -> 12 - 5 = 7
3) got 8 -> 7 + 8 = 15
A: 15

Q: A box had 20 pens, gave away 7, bought 4 more. How many now?
Let's think step by step:"""

for name, p in [("ZERO-SHOT", zero_shot), ("FEW-SHOT", few_shot), ("CHAIN-OF-THOUGHT", cot)]:
    print(f"===== {name} =====\n{p}\n")
print("few-shot teaches the format by example; chain-of-thought elicits reasoning before the answer.")
''')

md(r"""
## Recap — generation control

| Strategy | Rule | Effect |
|----------|------|--------|
| Greedy | argmax | deterministic, repetitive |
| Temperature `T` | softmax(logits/T) | creativity dial |
| Top-k | keep k best, renormalise | bans junk, fixed budget |
| Top-p (nucleus) | keep smallest set ≥ p | adaptive, modern default |
| Prompting | zero/few-shot, CoT, roles | steer without retraining |

## Common mistakes
1. **High temperature for factual tasks.** Use low `T` (or greedy) for facts/code; high `T` for brainstorming.
2. **Combining top-k and top-p without understanding.** They can be stacked, but know which one is binding.
3. **Forgetting greedy is deterministic.** If you need variety, you must sample (T>0, top-k/p).
4. **Vague prompts.** Specify format, role, and constraints; show examples (few-shot) for tricky formats.
5. **Skipping chain-of-thought on reasoning tasks** — letting the model "show work" markedly improves accuracy.

## Exercises (do them in new code cells)
1. Apply temperature `T=0.1` to logits `[2,1,0]` by hand. How close to greedy does it get?
2. Top-k with `k=3` on `[0.5,0.3,0.15,0.05]` — what are the renormalised probs?
3. Top-p with `p=0.6` on the same probs — which tokens survive? Compute by hand.
4. Write a few-shot prompt that teaches the model to translate English words to French.
5. Why does chain-of-thought help arithmetic, given the model predicts one token at a time?

---

**We can now set the mind's mood and brief at will — but a sobering question remains.** With all these
dials, is the mind *truly* smart, or just convincing? And can we *trust* it? Next:
[P27 — Evaluation, Alignment & Safety](P27_Evaluation_and_Safety.ipynb) — how we actually measure
whether a mind is good, and how we keep it safe.
""")

build("P26_Prompt_and_Decoding", execute=False)
