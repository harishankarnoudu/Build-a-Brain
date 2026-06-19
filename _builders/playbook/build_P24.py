"""Builder for playbook/P24_RLHF.ipynb

Aligning a model with human preferences. The 3-stage RLHF pipeline (SFT -> reward
model -> RL), the Bradley-Terry preference model and reward loss by hand, the PPO
idea with its KL leash, and DPO (the modern, RL-free alternative) with its loss
worked out by hand and verified. Connects to notebooks/13. TRL code read-only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P24 — RLHF: Reward Models, PPO & DPO  *(teach it taste — what people actually prefer)*

> **The story so far.** The mind has manners (P22) and we can afford to tune it (P23). But manners are
> the easy part of growing up. The hard part is *taste* — the thousand small judgements a parent can
> never quite put into words: be helpful but not preachy, honest but kind, confident but not
> arrogant. You can't hand a child a worksheet of "correct" answers for that. You can only react —
> "I liked how you handled that better than the other way" — over and over, until the judgement sinks
> in. This chapter is how the mind learns *taste*.

SFT (P22) teaches the mind to follow instructions, but among many *valid* answers, which is *better*?
More helpful, more honest, less harmful? Those preferences are nearly impossible to write as labelled
targets — yet a person can glance at two answers and instantly say "this one's better." **RLHF —
Reinforcement Learning from Human Feedback** — turns a pile of those pairwise judgements into a
training signal, and is a big part of what made ChatGPT-style assistants feel genuinely *aligned*
rather than merely fluent.

We cover the classic 3-stage pipeline, derive the **reward model** loss from the Bradley-Terry
preference model by hand, explain **PPO** and its crucial KL leash, then work the modern **DPO** loss
(which skips RL entirely) by hand and verify it. This is the heart of `notebooks/13`'s alignment
section.
""")

md(r"""
## B1 — The 3-stage RLHF pipeline

Teaching taste happens in three moves, like raising a critic: first the apprentice learns to do the
work at all, then it learns to *recognise* good work, then it practises until its own work earns the
nod.

```
Stage 1  SFT            : instruction-tune a base model (P22) -> a decent assistant π
Stage 2  Reward model   : collect human pairwise comparisons (answer A > answer B);
                          train a model r(prompt, answer) -> scalar "how good"
Stage 3  RL (PPO)       : fine-tune π to maximise the reward r, while a KL penalty
                          keeps it from drifting too far from the SFT model
```
The reward model *learns human taste* from comparisons; the RL step *optimises the assistant* to
score well by that taste. DPO (B4) later collapses stages 2–3 into one simple loss.
""")

md(r"""
## B2 — The reward model: learning preference from comparisons (Bradley-Terry, by hand)

Ask someone "rate this answer from 1 to 10" and you'll get noise — nobody agrees what a 7 means. Ask
instead "which of these two is better?" and people are sharp and consistent. So that's what we
collect: pairwise verdicts. Humans don't give numeric scores; they pick the better of two answers.
The **Bradley-Terry** model — the same maths that ranks chess players from match outcomes — turns a
pair of scalar rewards into the probability that one answer is preferred:
```
P(answer_w preferred over answer_l) = σ( r_w − r_l )        (σ = sigmoid, P07)
```
We train `r` to make preferred answers score higher, by **minimising** `−ln σ(r_w − r_l)` over all
human comparisons (`w` = winner, `l` = loser). It's binary cross-entropy (P07) on the reward gap.

**Worked example by hand.** If the model currently scores winner `r_w = 2.0`, loser `r_l = 1.0`:
```
gap     = r_w − r_l = 1.0
P(prefer w) = σ(1.0) = 0.731
loss    = −ln(0.731) = 0.313     (already > 0.5, so the model mostly agrees; loss is small)
```
If it had them backwards (`r_w=1, r_l=2`): `σ(−1)=0.269`, `loss = −ln(0.269)=1.31` — a big penalty.
""")

code(r'''
import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))
def reward_loss(r_w, r_l): return -np.log(sigmoid(r_w - r_l))

print("model agrees (r_w=2 > r_l=1):")
print("  P(prefer winner) = σ(1.0) =", round(sigmoid(1.0), 3), " (hand 0.731)")
print("  loss = -ln(0.731) =", round(reward_loss(2.0, 1.0), 3), " (hand 0.313, small)")
print("\nmodel disagrees (r_w=1 < r_l=2):")
print("  loss = -ln(σ(-1)) =", round(reward_loss(1.0, 2.0), 3), " (hand 1.31, big penalty)")
print("\ntraining lowers this loss -> reward model learns to score human-preferred answers higher.")
''')

md(r"""
## B3 — PPO: optimise the policy, but keep it on a leash

Now the mind practises — and immediately discovers the loophole every clever student finds: it's
easier to game the grader than to genuinely improve. Tell a kid "more words = better essay" and
you'll get a thousand words of padding. With a reward model in hand, **PPO (Proximal Policy
Optimization)** fine-tunes the assistant (the "policy" `π`) to generate answers the reward model
scores highly — and we have to keep it on a *leash* so it doesn't wander off into nonsense that merely
*looks* high-scoring. The objective, conceptually:
```
maximise   E[ reward(answer) ]  −  β · KL( π  ‖  π_SFT )
                 (score high)        (don't drift too far from the SFT model — P03's KL)
```
The **KL penalty** is the safety leash: without it the policy "hacks" the reward model — finding
weird high-scoring gibberish (**reward hacking**). The KL term keeps answers close to the sensible
SFT model. PPO works but is finicky: it needs the reward model, the policy, a reference model, and a
value model all in memory, with delicate tuning.
""")

code(r'''
import numpy as np
# illustrate the leash: total objective = reward - beta*KL. Too-high reward with huge KL is penalised.
def objective(reward, kl, beta): return reward - beta*kl
print("candidate answers (reward from RM, KL from the SFT model):")
for name, reward, kl in [("sensible answer", 3.0, 0.2),
                         ("slightly better-worded", 3.5, 0.6),
                         ("reward-hacking gibberish", 8.0, 9.0)]:
    print(f"  {name:>26}: reward={reward}, KL={kl} -> objective(β=0.5)={objective(reward,kl,0.5):+.2f}")
print("\nthe gibberish scores huge reward but its massive KL sinks the objective -> the leash holds.")
''')

md(r"""
## B4 — DPO: the same goal, no RL, no reward model (the modern default)

PPO works, but it's a circus: a policy, a reward model, a reference model, and a value model all juggled
in memory at once, with tuning so delicate it sometimes collapses for no obvious reason. Then someone
asked: do we even *need* the separate critic and the RL loop? What if the mind could learn taste
*directly* from the pairs of "this one's better"? **Direct Preference Optimization (DPO)** is the
breakthrough that made preference tuning simple. It
proves you can skip the separate reward model *and* the RL loop, and instead train the policy
directly on the preference pairs with a single classification-style loss:
```
loss = −ln σ( β·[ (log π(w) − log π_ref(w)) − (log π(l) − log π_ref(l)) ] )
```
In words: raise the policy's log-probability of the **winning** answer and lower it for the
**losing** one, *relative to the frozen reference (SFT) model* — with `β` controlling how far it may
move (the KL leash is baked in). It's the Bradley-Terry loss (B2) applied to the policy's own
log-prob ratios. Same alignment goal, one stable loss, far less machinery — which is why DPO largely
replaced PPO for open models.

**Worked example by hand.** Suppose the log-prob *ratios* vs the reference are `Δ_w = 0.5` for the
winner and `Δ_l = −0.3` for the loser, with `β = 1`:
```
margin = β·(Δ_w − Δ_l) = 1·(0.5 − (−0.3)) = 0.8
loss   = −ln σ(0.8) = −ln(0.690) = 0.371
```
Lowering this loss pushes the winner's ratio up and the loser's down.
""")

code(r'''
import numpy as np
def sigmoid(z): return 1/(1+np.exp(-z))

def dpo_loss(logp_w, logp_l, ref_w, ref_l, beta=1.0):
    margin = beta * ((logp_w - ref_w) - (logp_l - ref_l))     # the bracket above
    return -np.log(sigmoid(margin)), margin

loss, margin = dpo_loss(logp_w=0.5+0.0, logp_l=-0.3+0.0, ref_w=0.0, ref_l=0.0, beta=1.0)
print("margin = β·(Δ_w − Δ_l) =", round(margin, 3), " (hand 0.8)")
print("DPO loss = -ln σ(0.8)  =", round(loss, 3), " (hand 0.371)")

print("\nas the policy learns to prefer the winner, the margin grows and the loss shrinks:")
for dw in [0.8, 1.5, 3.0]:
    l, m = dpo_loss(dw, -0.3, 0.0, 0.0)
    print(f"  Δ_w={dw}: margin={m:.2f}  loss={l:.3f}")
''')

readonly(r'''
# DPO with HuggingFace TRL (read-only — trl is blocked here):
from trl import DPOTrainer, DPOConfig
# dataset rows: {"prompt":..., "chosen": <preferred answer>, "rejected": <worse answer>}
trainer = DPOTrainer(
    model=policy_model,            # the SFT model we are aligning
    ref_model=frozen_sft_model,    # π_ref in the loss above (kept fixed)
    args=DPOConfig(beta=0.1, learning_rate=5e-6),   # beta = the KL leash strength
    train_dataset=preference_dataset,
)
# trainer.train()   # one loss, no reward model, no PPO machinery
''')

md(r"""
## Recap — alignment with human preferences

| Stage / method | What it does |
|----------------|--------------|
| SFT (P22) | base behaviour: follow instructions |
| Reward model | learn human taste from pairwise comparisons (Bradley-Terry, B2) |
| PPO | RL-optimise policy for reward, KL-leashed to avoid reward hacking |
| **DPO** | same preference goal as one stable loss — no RM, no RL (B4) |
| `β` (both) | how far the model may drift from the reference |

## Common mistakes
1. **No KL leash.** The policy reward-hacks into gibberish. PPO needs `β·KL`; DPO bakes it into `β`.
2. **Confusing the reward model with the policy.** The RM scores answers; the policy generates them.
3. **Treating preferences as absolute scores.** They're *relative* (A vs B) — that's why Bradley-Terry uses the gap `r_w − r_l`.
4. **Forgetting the frozen reference in DPO.** The loss is about log-prob *ratios* vs `π_ref`, not raw log-probs.
5. **Reaching for PPO by default.** For most open-model work, DPO is simpler and as effective.

## Exercises (do them in new code cells)
1. Reward loss when `r_w = r_l` (model is indifferent): compute `−ln σ(0)` by hand. What does it equal and why?
2. In B3, find a `β` large enough that the "slightly better-worded" answer beats the "sensible" one.
3. DPO loss with `Δ_w = −0.2, Δ_l = 0.4, β = 1`: compute the margin and loss by hand. Is the model currently wrong?
4. Why does DPO need a frozen reference model? What would happen without it?
5. Explain reward hacking with a real-world analogy (e.g. optimising a metric that isn't quite the goal).

---

**The mind now has manners *and* taste — but it still only knows what it was born knowing, and it can
only talk, not act.** Ask it about yesterday's news and it will confidently make something up. Next:
[P25 — RAG, Tool Use & Agents](P25_RAG_and_Agents.ipynb) — give it a *library* so it stops inventing
facts, then *hands* so it can actually do things in the world.
""")

build("P24_RLHF", execute=False)
