# STORY_GUIDE.md — "Raising a Mind"

The single narrative spine for the whole curriculum (webapp **and** the P00–P29
notebooks). Everything is told as **one continuous story**: we are *raising a
mind* — from a single bare number into a thinking machine. Use this file as the
voice spec so every chapter, figure, and notebook stays in the same key.

---

## The arc (one mind, growing up)

The reader is not "studying topics." They are **raising a creature** and watching
it wake up, stage by stage. Each part is a chapter of its life:

| Part | Life stage | What the mind gains |
|------|-----------|---------------------|
| **0 · Foundations** | *Born* | It's just numbers. It learns the alphabet of thought: numbers in motion, direction (dot product), space it can bend (matrices), the *urge to get better* (gradient descent), how to weigh uncertainty (probability), and its first meal (data). |
| **1 · Classical ML** | *First instincts* | Before neurons: simple reflexes. Draw the line that explains the world (regression), make a yes/no call (logistic), learn humility — don't just memorise (overfitting/evaluation), and a starter kit of instincts (the algorithms tour). |
| **2 · Deep Learning** | *A brain forms* | Wire many tiny decisions into neurons. The flash of learning (backprop). Learning to learn faster (optimizers). Tricks that keep a deep brain from falling apart (init/norm/dropout/residuals). A living lab to play in (playground). Then eyes (CNNs) and memory (RNNs). |
| **3 · Language** | *Learning to read & mean* | The clumsy old way it read by counting (classical NLP). Words become *places* in space, meaning becomes direction (embeddings). It learns to **pay attention** (seq2seq → attention). |
| **4 · Transformers** | *Its true form* | Snap every piece together into the architecture behind modern AI — and hand off to TinyGPT. |
| **5 · LLM Engineering** | *Raising a real mind* | Feed it the whole internet (pretraining/scaling). Teach it manners (SFT). Reshape a giant cheaply (LoRA/quantization). Teach it judgement — what people actually prefer (RLHF/DPO). Give it a library and hands (RAG/tools/agents). Tune its voice (decoding). Test it and keep it safe (eval/safety). |
| **6 · Deployment** | *Out into the world* | Make it fast enough to serve millions (serving/KV-cache). Keep it alive, healthy, and honest forever (MLOps). |

Recurring through-lines to call back to (so the story feels like ONE story):
- **"predict → measure how wrong → step downhill"** is the *heartbeat* introduced
  in Part 0 and re-met in regression, neural nets, and pretraining. Always name it.
- **The dot product** is the mind's single primitive sense ("how much do two
  things agree?"). It returns in neurons, attention, and retrieval.
- **The same training loop, a billion-fold bigger** is the punchline of Part 5.

---

## Voice rules (apply everywhere)

1. **Open on a scene, not a definition.** Start each chapter with a concrete,
   real-world moment the reader already knows (a thermostat nudging the heat, a
   spam folder, autocomplete finishing your sentence, a toddler sorting shapes).
   The math is the *explanation of the scene*, never the opening.
2. **Why before what before how.** Always: *why does the mind need this?* →
   *what is the idea, in plain words?* → *how, by hand on tiny numbers?* → verify.
   Keep the existing "by hand on tiny numbers, then code that re-computes it"
   teaching style — story does **not** replace rigor, it *delivers* it.
3. **Talk to the mind like it's alive.** "Now the mind can…", "It's tempted
   to cheat by memorising…", "Give it eyes." Second person for the reader
   ("you"), affectionate third person for the machine ("it").
4. **One idea per chapter, named like a plot beat.** e.g. *The flash of learning*
   (backprop), *The temptation to memorise* (overfitting), *Meaning becomes
   direction* (embeddings).
5. **Demote heavy math.** Lead with intuition + a hand example; put dense
   formulas behind `ShowMath` (webapp) or below the worked example (notebooks).
   Math must still be **correct** and verifiable.
6. **Every figure is a playground with a mission.** Don't say "here is a figure."
   Say what to *try* and what you'll *feel happen*. Give 2–3 concrete missions
   ("make a·b negative", "push the learning rate until it explodes").
7. **End each chapter looking forward.** A one-line bridge to the next life
   stage, so the reader wants to turn the page.
8. **Fun, not cute.** Vivid and warm, but never sacrifice correctness or talk
   down. A smart friend explaining at a whiteboard, not a children's book.

## Quick phrase bank (reuse for continuity)
- "the heartbeat of all learning: predict → measure → step downhill"
- "direction becomes a single number"
- "the mind's first temptation: memorise instead of understand"
- "wire many tiny decisions together and a brain appears"
- "meaning becomes geometry"
- "the same loop, a billion-fold bigger"
