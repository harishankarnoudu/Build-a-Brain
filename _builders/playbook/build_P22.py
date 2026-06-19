"""Builder for playbook/P22_Supervised_FineTuning.ipynb

How a raw next-token model becomes an instruction-following assistant. The SFT
objective (same cross-entropy, but on prompt->response pairs, with the prompt
tokens masked from the loss), chat templates, and why only completion tokens are
trained. Tiny runnable masking demo; HF SFT read-only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P22 — Supervised Fine-Tuning & Instruction Tuning  *(finishing school: teach the brilliant mumbler manners)*

> **The story so far.** In P21 we fed the mind the whole internet and watched something extraordinary
> happen: out of pure next-token prediction — the same *predict → measure → step downhill* heartbeat
> from day one, just a billion-fold bigger — it learned grammar, facts, and reasoning all at once. But
> the creature that walked out of pretraining is a **brilliant mumbler**. It knows astonishing things,
> yet it doesn't know how to *behave*. Now it leaves the nursery and enters **finishing school**: we
> teach it manners.

Picture asking a freakishly well-read but socially clueless prodigy "What is the capital of France?"
A freshly pretrained model (P21) is a brilliant **autocomplete** — it continues text — but it
doesn't *follow instructions*. So it might continue with *more questions* ("What is the capital of
Germany? What is the capital of…"), because on the open web a question is usually followed by more
questions, not by an answer. **Supervised Fine-Tuning (SFT)**, a.k.a. **instruction tuning**, is the
etiquette class: we keep training the model, but now on curated **(instruction → ideal response)**
pairs, so it learns the *behaviour* of answering.

The beautiful part — and the through-line of this whole part of the journey: it's the **same
next-token cross-entropy** (P03) you already know, the same loop, just pointed at better data — with
one twist (mask the prompt out of the loss). This is `notebooks/12_SecurityGPT` and the first half of
`notebooks/13`, generalised.
""")

md(r"""
## B1 — Base model vs instruction-tuned: the behaviour gap

You've met this person: someone who has read every book in the library but, asked a simple question
at dinner, launches into a related tangent instead of just answering. That's a base model. The mind
hasn't gotten *dumber* in finishing school — it's exactly the same network, same weights-shaped brain.
We only changed the example it imitates. Same architecture, different *training data*, hence
different behaviour:
```
prompt: "What is the capital of France?"
BASE model (pretrained on raw web): "What is the capital of Germany? What is the capital of..."
                                     (continues the *pattern* of question lists)
SFT model (tuned on instruction/answer pairs): "The capital of France is Paris."
                                     (learned to *answer*)
```
Nothing about the network changed — only what we trained it to predict. SFT teaches the *format and
intent* of being helpful.
""")

md(r"""
## B2 — The SFT data format: prompt → response, with a chat template

Before a child can hold a conversation it needs to learn whose *turn* it is — when someone is asking
and when it's their job to reply. The mind needs the same scaffolding. SFT data is pairs of
`(instruction, response)`. To train, we wrap them in a **chat template** — special tokens marking who
said what — so the model learns turn structure. A simplified template:
```
<|user|> What is 2+2? <|assistant|> 4 <|end|>
```
At inference you feed everything up to `<|assistant|>` and let the model generate the response.
Real templates (ChatML, Llama, etc.) differ in exact tokens but do the same job.
""")

code(r'''
# format a few instruction/response pairs with a simple chat template
pairs = [
    ("What is 2+2?", "4"),
    ("Capital of France?", "Paris"),
    ("Say hello", "Hello! How can I help you?"),
]
def apply_template(instruction, response=None):
    s = f"<|user|> {instruction} <|assistant|>"
    return s + (f" {response} <|end|>" if response is not None else "")

for ins, resp in pairs:
    print(apply_template(ins, resp))
print("\nat inference you send up to '<|assistant|>' and generate the rest:")
print(apply_template("Capital of Japan?"))
''')

md(r"""
## B3 — The key twist: mask the prompt out of the loss

A good tutor doesn't grade the student on repeating the question back — only on the *answer* they
give. We want exactly that: the mind should learn to *produce the response*, not to predict the
user's prompt (that's given to it for free). So during SFT we compute cross-entropy **only on the
response tokens** and **mask** (ignore) the prompt tokens. PyTorch uses the label value `-100` to
mean "ignore this position in the loss."

**Worked example by hand.** Sequence tokens with a 3-token prompt and 2-token response:
```
tokens : [<user>, Q, <asst>,  A1, A2, <end>]
labels : [ -100, -100, -100,  A1, A2, <end>]   # prompt positions -> -100 (ignored)
```
Only the `A1, A2, <end>` positions contribute to the loss, so gradient only teaches the response.
""")

code(r'''
import torch, torch.nn as nn
# tiny demo: loss computed over all tokens vs only the response tokens
V = 10
logits = torch.randn(6, V)                      # 6 positions, vocab 10
targets_full   = torch.tensor([2, 3, 4, 5, 6, 7])           # pretend token ids
targets_masked = torch.tensor([-100, -100, -100, 5, 6, 7])  # prompt (first 3) ignored

loss_full   = nn.functional.cross_entropy(logits, targets_full)
loss_masked = nn.functional.cross_entropy(logits, targets_masked, ignore_index=-100)
print("loss over ALL tokens (wrong for SFT) :", round(loss_full.item(), 3))
print("loss over RESPONSE tokens only (SFT) :", round(loss_masked.item(), 3))
print("-100 positions are skipped -> the model is trained ONLY to generate the answer.")
''')

md(r"""
## B4 — The SFT training loop is ordinary fine-tuning

Here's the reassuring secret: finishing school uses no new machinery. With templated, masked data,
SFT is *just* the training loop from P11/P14 — the same heartbeat — only with a small learning rate,
because we're gently nudging an already-capable mind, not raising a new one from scratch. A few epochs
over thousands of high-quality examples is often enough — quality and diversity of the examples
matter far more than quantity (a handful of impeccable role models beats a crowd of mediocre ones).
This is what produces the "-instruct" / "-chat" models you actually talk to.
""")

readonly(r'''
# SFT with HuggingFace TRL's SFTTrainer (read-only — trl/transformers blocked here):
from trl import SFTTrainer, SFTConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")   # a PRETRAINED base model
tok   = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
# dataset rows look like {"messages": [{"role":"user",...},{"role":"assistant",...}]}
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(learning_rate=2e-5, num_train_epochs=3,        # small LR: gentle nudge (P12)
                   completion_only_loss=True),                    # == the B3 prompt masking!
    train_dataset=instruction_dataset,
)
# trainer.train()   # then save -> you have an instruction-following assistant
''')

md(r"""
## Recap — supervised fine-tuning

| Aspect | SFT |
|--------|-----|
| Goal | teach instruction-following *behaviour* |
| Data | curated (instruction → response) pairs |
| Format | chat template with role tokens |
| Loss | cross-entropy **on response tokens only** (mask prompt with −100) |
| Learning rate | small (gentle nudge of a capable model) |
| Result | "-instruct"/"-chat" assistant |

SFT gets you a polite, helpful mind that answers the question. But there's a subtler lesson manners
can't teach: among several *valid* answers, which one is genuinely *better*? That's taste, and you
can't write it down as a target — you can only show the mind what people prefer. We'll teach it that
soon (RLHF/DPO, P24). First, though, a practical problem: our mind is now *enormous*, and finishing
school for a giant is expensive. The next chapter shows how to reshape a mountain on a laptop.

## Common mistakes
1. **Training on the prompt tokens.** Always mask them (−100) — otherwise you teach the model to parrot questions.
2. **Too-high learning rate.** It erases pretrained knowledge ("catastrophic forgetting"). Keep it small.
3. **Low-quality or repetitive data.** A few thousand great examples beat millions of mediocre ones.
4. **Mismatched chat template** between training and inference → the model behaves oddly. Use the *same* template.
5. **Expecting SFT to fix subtle preferences** (tone, harmlessness, ranking good vs better). That's RLHF/DPO (P24).

## Exercises (do them in new code cells)
1. Template the pair `("Translate 'hi' to French", "salut")` with the B2 function.
2. Build the label mask for a sequence with a 4-token prompt and 3-token response (which positions are −100?).
3. In B3, recompute `loss_masked` if *all* labels are −100. What happens, and why is that a bug to guard against?
4. Why does a tiny learning rate matter more for SFT than for pretraining?
5. Give a real example where SFT alone is insufficient and you'd want preference tuning.

---

**The mind has manners now — but it's a giant, and tutoring a giant is costly.** Next:
[P23 — PEFT: LoRA / QLoRA / Quantization](P23_LoRA_and_Quantization.ipynb) — how to reshape a huge
mind on a single laptop GPU by adding tiny *footpaths* instead of moving the whole mountain, with the
low-rank math worked out by hand.
""")

build("P22_Supervised_FineTuning", execute=False)
