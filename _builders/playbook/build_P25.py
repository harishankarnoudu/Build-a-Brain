"""Builder for playbook/P25_RAG_and_Agents.ipynb

Giving a model external knowledge and actions. Embedding-based retrieval (cosine
similarity from P01/P18) built from scratch as a tiny vector search, the RAG
prompt-augmentation pattern, then tool use and the agent loop. Runnable retrieval
demo; framework code read-only.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, readonly, reset, build

reset()

md(r"""
# P25 — RAG, Tool Use & Agents  *(give it a library so it stops making things up, then hands so it can act)*

> **The story so far.** The mind has manners (P22) and taste (P24) — but it's been raised entirely on
> books that stopped printing on some cutoff date, and it has no way to look anything up. So when you
> ask about something it never read, it does the most human thing imaginable: it **makes something up**,
> fluently and with total confidence. A polished, tasteful mind that hallucinates is dangerous. This
> chapter does two things: it walks the mind into a **library** so it can answer from sources instead
> of from fog, and then it gives it **hands** — tools it can pick up and use to act in the world.

An LLM only knows what was in its training data, frozen at a cutoff date, and it can't look anything
up — so it sometimes **hallucinates** confident-but-wrong answers. **RAG (Retrieval-Augmented
Generation)** fixes the knowledge gap: *retrieve* the relevant books from the library and lay them
open on the desk (paste them into the prompt) so the mind answers *from sources*. **Tool use** and
**agents** go further: hand the mind a calculator, a search box, a code runner — let it call functions
and act in loops.

And here the journey loops back to its very first sense. The engine of the library is **embedding
similarity search** — exactly the **dot product** / cosine similarity from P01/P18, the mind's
original primitive feeling of "how much do two things agree?" We build a tiny vector search from
scratch (fully runnable), then show the RAG and agent patterns.
""")

md(r"""
## B1 — Embedding search: find the most relevant document (from scratch)

A good librarian doesn't match your question word-for-word against the shelves — they understand what
you *mean* and walk straight to the right aisle. The mind's librarian does this with geometry: turn
every document and your question into a point in space, then fetch whatever points *nearest* in
meaning. RAG's retriever works like this:
1. **Embed** every document into a vector (P18) and store them.
2. **Embed the query** the same way.
3. **Rank** documents by **cosine similarity** (P01 B6) to the query; return the top matches.

We use tiny hand-made embeddings so the ranking is verifiable. The query about "pets" should match
the animal document, not the finance one — by geometry alone.

**Worked example by hand.** Query `q=[1,0]`, docs `d0=[0.9,0.1]` (pets), `d1=[0.1,0.9]` (finance):
```
cos(q,d0) = (1·0.9+0·0.1)/(1·sqrt(0.82)) = 0.9/0.906 = 0.993   <- top match
cos(q,d1) = (1·0.1+0·0.9)/(1·sqrt(0.82)) = 0.1/0.906 = 0.110
```
""")

code(r'''
import numpy as np
from numpy.linalg import norm
def cos(a, b): return (a @ b) / (norm(a)*norm(b) + 1e-9)

# tiny "knowledge base": each doc has text + a 2-D embedding ([pet-ness, finance-ness])
docs = [
    ("Cats and dogs are popular pets.",       np.array([0.9, 0.1])),
    ("Stocks and bonds are investments.",     np.array([0.1, 0.9])),
    ("A kitten is a young cat.",              np.array([0.8, 0.2])),
]
query = ("What animals make good pets?", np.array([1.0, 0.0]))

scored = sorted(docs, key=lambda d: cos(query[1], d[1]), reverse=True)
print("query:", query[0], "\nranked documents by cosine similarity:")
for text, emb in scored:
    print(f"  {cos(query[1], emb):.3f}  {text}")
print("\ntop match is the pet document — retrieval by geometry, no keywords needed.")
''')

md(r"""
## B2 — The RAG pattern: retrieve, then stuff the prompt

The librarian has pulled the right books; now they lay them open on the desk and say "answer from
*these*." Once you have the top documents, RAG simply **inserts them into the prompt** as context and
asks the model to answer using them. The mind now answers from *provided sources* it can quote,
instead of from fuzzy memory — which slashes hallucination and lets you cite references and use
fresh/private data the model never trained on.
```
[retrieved context]
Use ONLY the context above to answer.
Question: <user question>
```
""")

code(r'''
import numpy as np
from numpy.linalg import norm
def cos(a,b): return (a@b)/(norm(a)*norm(b)+1e-9)
docs = [("Cats and dogs are popular pets.", np.array([0.9,0.1])),
        ("Stocks and bonds are investments.", np.array([0.1,0.9])),
        ("A kitten is a young cat.", np.array([0.8,0.2]))]
q_text, q_emb = "What animals make good pets?", np.array([1.0,0.0])

top = sorted(docs, key=lambda d: cos(q_emb, d[1]), reverse=True)[:2]
context = "\n".join(f"- {t}" for t,_ in top)
prompt = f"""Context:
{context}

Use ONLY the context above to answer.
Question: {q_text}
Answer:"""
print(prompt)
print("\n<- this augmented prompt goes to the LLM. It answers from the retrieved facts, with sources.")
''')

md(r"""
## B3 — Tool use: let the model call functions

A library answers questions; it can't *do* anything. For that the mind needs hands. Oddly, a model is
brilliant at language but unreliable at multiplying two three-digit numbers — so instead of forcing it
to compute, we let it *reach for a calculator*. A model can't reliably do arithmetic or fetch live
data — but it can **emit a request to call a tool**, which your code executes and feeds back. The
pattern: describe available tools; the model outputs a structured call (e.g. JSON); you run it and
return the result; the model continues with the answer. We simulate one round with a calculator tool.
""")

code(r'''
import json
# tools your code exposes to the model
def calculator(expression):
    # safe arithmetic eval: only digits/operators allowed (never use raw eval on model output)
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed: return "error: invalid characters"
    import ast, operator as op
    ops = {ast.Add:op.add, ast.Sub:op.sub, ast.Mult:op.mul, ast.Div:op.truediv, ast.USub:op.neg}
    def ev(n):
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.BinOp): return ops[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp): return ops[type(n.op)](ev(n.operand))
        raise ValueError
    return ev(ast.parse(expression, mode="eval").body)

# the model (simulated) decides it needs the calculator and emits a structured call:
model_output = '{"tool": "calculator", "expression": "123 * 47"}'
call = json.loads(model_output)
result = calculator(call["expression"])
print("model asked to call:", call)
print("tool result        :", result)
print("...which is fed back so the model can answer: '123 * 47 = 5781'")
''')

md(r"""
## B4 — Agents: the perceive → think → act loop

One tool call is a single reach. But real tasks take several moves in a row — look something up,
*then* calculate, *then* summarise — each step deciding the next, the way you'd plan a trip rather
than buy one ticket. An **agent** wraps tool use in a loop so the mind can take *multiple* steps
toward a goal:
```
loop:
   think  : model reasons about what to do next
   act    : model picks a tool + arguments
   observe: your code runs the tool, returns the result
   until the model decides it has the final answer
```
This lets a model break a hard task into steps — search, then calculate, then summarise. The risks
(loops that never end, wrong tool calls, compounding errors) are why agents need step limits,
validation, and guardrails (P27). We simulate a 2-step agent.
""")

code(r'''
# a toy agent loop: goal needs a lookup THEN a calculation
def search(q):  return {"price_per_unit": 20} if "price" in q else {}
def calc(a, b): return a * b

steps = [
    {"think": "I need the unit price first", "act": ("search", "price of widget")},
    {"think": "Now multiply price by quantity 3", "act": ("calc", None)},
]
memory = {}
for i, step in enumerate(steps, 1):
    print(f"step {i}: {step['think']}")
    tool, arg = step["act"]
    if tool == "search":
        memory["price"] = search(arg)["price_per_unit"]; print(f"   observed price = {memory['price']}")
    elif tool == "calc":
        ans = calc(memory["price"], 3); print(f"   computed total = {ans}")
print("\nfinal answer: 3 widgets cost", calc(memory["price"], 3))
print("the agent chained search -> calculate to solve what one LLM call could not.")
''')

readonly(r'''
# RAG/agents with a framework (read-only — these libs aren't installed here):
# (1) embeddings + vector store
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_vecs = embedder.encode(documents)          # real semantic embeddings (P18 at scale)
# store in a vector DB (FAISS, Chroma, etc.) and query by cosine — exactly B1, scaled.

# (2) an agent with tools (LangChain-style sketch)
# from langchain.agents import initialize_agent, Tool
# tools = [Tool(name="calculator", func=calculator, description="do math")]
# agent = initialize_agent(tools, llm); agent.run("What is 17% of 240, plus 12?")
''')

md(r"""
## Recap — beyond the model's own memory

| Capability | Mechanism | Built on |
|------------|-----------|----------|
| Retrieval | embed + cosine top-k (B1) | P01, P18 |
| RAG | retrieve → stuff prompt → answer from sources | B2 |
| Tool use | model emits call → your code runs it → feed back | B3 |
| Agent | loop think/act/observe to a goal | B4 |

## Common mistakes
1. **Stuffing too much context.** Retrieve the *most relevant* few chunks; irrelevant context hurts and costs tokens.
2. **No "use only the context" instruction** → the model ignores sources and hallucinates anyway.
3. **Running model-generated code/SQL/`eval` blindly.** Validate and sandbox tool inputs (note our calculator whitelist).
4. **Agents with no step limit.** They can loop forever or rack up cost; cap iterations.
5. **Assuming RAG fixes everything.** Bad retrieval = bad answers; the retriever's quality is the ceiling.

## Exercises (do them in new code cells)
1. Add a doc `("Bonds pay fixed interest.", [0.05, 0.95])` and a finance query. Does it rank above the pet docs?
2. Compute `cos([1,1], [2,2])` by hand. Why does direction (not length) decide the ranking?
3. Extend the B3 calculator to reject `"__import__('os')"`. Why is the whitelist essential?
4. Add a third agent step in B4 that applies a 10% discount to the total.
5. Give a real task where RAG is the right tool, and one where an agent (multi-step) is required.

---

**The mind can now look things up and act — give it a library and hands.** But its actual *voice* —
whether it answers crisply or rambles, plays it safe or takes creative risks — is set by a few hidden
dials we haven't touched yet. Next:
[P26 — Prompt Engineering & Decoding](P26_Prompt_and_Decoding.ipynb) — the dials that set the mind's
mood: how it turns probabilities into words (greedy, temperature, top-k, top-p) and how the way you
phrase a request steers what comes back.
""")

build("P25_RAG_and_Agents", execute=False)
