"""Builder for playbook/P28_Serving_and_Inference.ipynb

How trained models are deployed fast and cheap. The KV-cache (with the redundant-
compute it removes shown by hand), continuous batching, quantized inference (recap
P23), and what vLLM/TGI do. Runnable KV-cache cost demo.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P28 — Serving & Inference Optimization  *(make it fast enough to answer millions)*

> **The story so far.** The mind is grown, wise, and trustworthy (P22–P27). It's leaving home — out
> into the world to actually *help people*. But raising it was a one-time effort; **answering** people
> is forever, and the moment a second person shows up, then a thousandth, then a millionth, a new
> problem appears that had nothing to do with intelligence: *speed*. A brilliant mind that takes a
> minute to reply, and costs a fortune per word, helps no one at scale. This chapter is about making
> the mind quick and cheap enough to talk to everyone at once.

Training a model is a one-time cost; **serving** it — answering millions of requests — is the
forever cost. Naive generation is wasteful (the mind re-reads the entire conversation from the top
before every single new word, like someone who re-reads the whole book each time they add a sentence),
so production systems use a handful of powerful tricks. This notebook explains the big ones — the
**KV-cache**, **continuous batching**, and **quantized inference** — and what tools like **vLLM** do.
We quantify the KV-cache's savings by hand so you see *why* it's non-negotiable.
""")

md(r"""
## B1 — Why naive generation is wasteful

Picture the mind writing an answer the naive way: to choose the 10th word it re-reads words 1 through
9; to choose the 11th it re-reads 1 through 10; and so on, re-processing the same growing pile of text
at every single step. It's the most wasteful possible way to write. A transformer generates one token
at a time, each time running the whole prompt-so-far through the model. Naively, to generate token `t` you recompute attention over all `t−1` previous tokens — and
you do that again for token `t+1`, re-processing the same earlier tokens *every step*. The repeated
work grows quadratically.

**Worked example by hand.** Generating a 100-token answer:
```
step 1 processes 1 token, step 2 processes 2, … step 100 processes 100
naive total = 1 + 2 + … + 100 = 100·101/2 = 5050 token-computations
```
Most of that is recomputing the *same* earlier tokens. The KV-cache removes the redundancy.
""")

code(r'''
n = 100
naive = sum(range(1, n+1))          # 1+2+...+n
print(f"naive generation of {n} tokens = {naive} token-computations (1+2+...+{n})")
print("the early tokens get reprocessed again and again — pure waste.")
''')

md(r"""
## B2 — The KV-cache: compute each token's Key/Value once

The fix is the obvious one: stop re-reading what hasn't changed. Jot down what you worked out about
each earlier word the first time you read it, and just glance at your notes afterward. That notebook
is the KV-cache. In attention (Notebook 00 B9), each token produces a **Key** and **Value**. A past token's K and V
**never change** as generation continues — so recomputing them every step is wasted work. The
**KV-cache** stores them: each new token computes *its own* K/V once, appends to the cache, and
attends over the cached K/V of all previous tokens.

**Worked example by hand.** Same 100-token generation, but each token's K/V computed once:
```
cached total = 100 token-computations (one per generated token)
naive total  = 5050
speedup      = 5050 / 100 ≈ 50×   (grows with sequence length)
```
This single optimisation is why interactive chat is fast. The cost: memory — the cache grows with
sequence length and batch size (often the real bottleneck in serving).
""")

code(r'''
n = 100
naive  = sum(range(1, n+1))
cached = n                          # each token's K/V computed exactly once
print(f"naive  : {naive} token-computations")
print(f"cached : {cached} token-computations")
print(f"speedup: {naive/cached:.0f}x  (and it grows with length)")
print("\ntradeoff: the KV-cache uses memory that grows with (sequence length x batch x layers).")
''')

md(r"""
## B3 — Continuous batching: keep the GPU busy

One mind, one reader is easy. The real challenge is the *crowd*: thousands of people all talking to
the mind at once, each mid-conversation, each finishing at a different moment. Handle them like a
short-order cook who plates each dish the instant it's ready and immediately starts the next ticket —
never letting the kitchen sit idle waiting for the slowest order. A GPU is most efficient processing
many sequences at once, but requests arrive at different times and finish at different lengths. **Continuous (in-flight) batching** dynamically adds new requests into
the batch as others finish, instead of waiting for a whole batch to complete. This keeps utilisation
high and throughput up — it's a scheduling trick, not a model change, and it's a core reason serving
engines are far faster than a naive loop.
""")

code(r'''
# illustrate the win: static batching wastes slots when sequences finish early
# 4 requests needing these many tokens; static batch waits for the longest each round
lengths = [5, 20, 8, 15]
static_rounds = max(lengths)                 # everyone waits for the slowest (20)
static_slot_use = sum(lengths) / (len(lengths)*static_rounds)
print("request lengths      :", lengths)
print(f"static batch: runs {static_rounds} rounds; GPU slot utilisation = {static_slot_use:.0%}")
print("continuous batching backfills finished slots with new requests -> utilisation near 100%")
''')

md(r"""
## B4 — Quantized inference and serving engines

Two last levers — and you've met the first one already, wearing a different hat. The coarse-number
trick we used in P23 to *shrink the mountain for training* works just as well to *speed it up for
answering*. And in practice nobody wires all this together by hand: there are off-the-shelf engines
that bundle every trick in this chapter behind a single clean interface. Two more levers for cheap
serving:
- **Quantized inference** — run the model in int8/int4 (P23 B3). ~4–8× less memory and faster on
  supported hardware, with little quality loss for inference. Lets a big model fit on a smaller GPU.
- **Serving engines** — **vLLM**, **TGI**, **TensorRT-LLM** bundle all these tricks (paged KV-cache,
  continuous batching, optimized kernels, quantization) behind an OpenAI-compatible API. You rarely
  hand-roll serving; you run one of these.

`PagedAttention` (vLLM's signature trick) manages the KV-cache like virtual memory pages, so memory
isn't wasted on padding — squeezing far more concurrent requests onto one GPU.
""")

readonly(r'''
# Serving with vLLM (read-only — not installed here; this is the typical usage):
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", quantization="awq")   # quantized (P23)
params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)        # decoding (P26)
outputs = llm.generate(["Explain attention in one sentence."], params)
# vLLM automatically does paged KV-cache (B2) + continuous batching (B3) across all requests.
# In production you'd start `vllm serve <model>` and hit its OpenAI-compatible HTTP endpoint.
''')

md(r"""
## Recap — fast, cheap inference

| Technique | Saves | Cost / note |
|-----------|-------|-------------|
| KV-cache | recomputing past tokens (≈50× @100 tok) | memory grows with length×batch |
| Continuous batching | idle GPU slots | scheduler complexity |
| Quantized inference | memory & latency (4–8×) | tiny quality loss |
| Serving engine (vLLM/TGI) | engineering effort | use it, don't hand-roll |
| PagedAttention | KV-cache memory waste | vLLM's key trick |

## Common mistakes
1. **Generating without a KV-cache** in production — needlessly ~50× slower.
2. **Ignoring KV-cache memory.** It, not compute, is often what caps your concurrent users / context length.
3. **Static batching under variable load** → wasted GPU; use continuous batching.
4. **Over-quantizing for inference and not measuring quality.** Usually fine, but verify on your task.
5. **Building your own server.** vLLM/TGI already implement all of this, well-tested.

## Exercises (do them in new code cells)
1. Compute naive vs KV-cached token-computations for a 50-token generation. What's the speedup?
2. At what sequence length does the KV-cache speedup exceed 100×? (Solve `n(n+1)/2 / n > 100`.)
3. For request lengths `[3, 30, 5]`, compute static-batch GPU slot utilisation by hand.
4. A 7B model in float16 is ~14 GB. Estimate its size in int4. Does it fit a 6 GB GPU?
5. Explain in one sentence why a past token's Key and Value never change during generation.

---

**The mind is out in the world now, answering millions, fast and cheap.** But shipping it isn't the
end — a living mind has to be *kept* alive: watched, maintained, and honestly held to account as the
world around it shifts. Next, the finale: [P29 — MLOps](P29_MLOps.ipynb) — keeping the mind you raised
alive, healthy, and honest, where the loop never closes and the journey comes full circle.
""")

build("P28_Serving_and_Inference", execute=False)
