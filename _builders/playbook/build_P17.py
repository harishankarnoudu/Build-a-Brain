"""Builder for playbook/P17_Classical_NLP.ipynb

Opens Part 3. Turning text into numbers the pre-embedding way: tokenization,
Bag-of-Words, TF-IDF (every count by hand), and an n-gram language model that
predicts the next word by counting — the conceptual ancestor of GPT. Sets up why
we need embeddings (P18) and attention.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _pbcommon import md, code, reset, build

reset()

md(r"""
# P17 — Classical NLP  *(the mind's clumsy first reading lessons)*

> **The story so far.** The mind has eyes (P15) and memory (P16), and it knows — from the way its
> memory faded — that it *wants* to read. So we open **Part 3: Language**, and sit it down for its
> very first reading lessons. Like any beginner, it starts clumsily: it can't grasp *meaning* yet, so
> it does the only thing it can — it **counts words**. It tallies which words appear, how often, and
> which word tends to follow which. It's the reading equivalent of sounding out letters one at a
> time: useful, honest, and destined to miss the point. Watching exactly *how* it misses the meaning
> is what makes the next chapters — where words finally gain meaning — feel inevitable.

Before neural embeddings (P18) and transformers, this is genuinely how machines read: turn text into
numbers by **counting**. We teach the mind four counting tricks — tokenization, **Bag-of-Words**,
**TF-IDF**, and **n-gram language models**. They're still fast, useful baselines, and — the punchline
— the n-gram model predicts the next word in *exactly* the spirit GPT does, only by counting instead
of understanding. See how counting works and the leap to a neural language model becomes obvious.

Every count is worked by hand on a tiny 3-sentence corpus so the matrices stay fully traceable. This
connects forward to `notebooks/01_Tokenizer` (which builds a real tokenizer).
""")

md(r"""
## B1 — Tokenization: splitting text into units

A child learning to read first chops the page into words. The mind must do the same — it can't
swallow a sentence whole, the way you can't swallow a sentence as one sound. It needs a list of
**tokens**, each mapped to an integer id. The simplest tokenizer splits on whitespace and lowercases.
The set of unique tokens is the mind's first **vocabulary**, and each token gets an id.

**Worked example by hand.** Corpus:
```
doc0: "the cat sat"
doc1: "the dog sat"
doc2: "the cat ran"
```
Unique tokens (sorted): `cat, dog, ran, sat, the` → ids `{cat:0, dog:1, ran:2, sat:3, the:4}`. So
`"the cat sat"` encodes to `[4, 0, 3]`.
""")

code(r'''
corpus = ["the cat sat", "the dog sat", "the cat ran"]
tokens = [doc.lower().split() for doc in corpus]
vocab = sorted({t for doc in tokens for t in doc})
stoi = {w: i for i, w in enumerate(vocab)}            # string -> id
print("tokens   :", tokens)
print("vocab    :", vocab)
print("stoi     :", stoi)
print("encode 'the cat sat' ->", [stoi[w] for w in "the cat sat".split()], " (hand [4,0,3])")
''')

md(r"""
## B2 — Bag-of-Words: a document as a vector of counts

Imagine tipping a sentence into a bag and shaking it — the words tumble together and the *order*
falls out. What's left is just a tally: which words, how many. That's **Bag-of-Words (BoW)**: it
represents a document by *how many times* each vocab word appears, order forgotten. Each document
becomes a vector of length = vocab size. Crude — but it's the mind's first move from text to numbers
it can actually compute on.

**Worked example by hand.** Vocab order `[cat, dog, ran, sat, the]`:
```
doc0 "the cat sat" -> cat:1 dog:0 ran:0 sat:1 the:1 -> [1,0,0,1,1]
doc1 "the dog sat" -> cat:0 dog:1 ran:0 sat:1 the:1 -> [0,1,0,1,1]
doc2 "the cat ran" -> cat:1 dog:0 ran:1 sat:0 the:1 -> [1,0,1,0,1]
```
Now text is a number matrix — feed it to any Part 1 classifier (spam detection, sentiment, …).
""")

code(r'''
import numpy as np
corpus = ["the cat sat", "the dog sat", "the cat ran"]
vocab = ["cat","dog","ran","sat","the"]; stoi = {w:i for i,w in enumerate(vocab)}
BoW = np.zeros((3, 5), int)
for d, doc in enumerate(corpus):
    for w in doc.split():
        BoW[d, stoi[w]] += 1
print("vocab :", vocab)
print("BoW matrix (rows=docs, cols=word counts):\n", BoW)
print("hand: [1,0,0,1,1] / [0,1,0,1,1] / [1,0,1,0,1]")
''')

md(r"""
## B3 — TF-IDF: down-weight words that are everywhere

The mind notices something: in every document, `"the"` shows up — so it tells us nothing about *which*
document we're looking at. It's like a word that everyone in the room is shouting; it carries no
signal. The fix is **TF-IDF**: multiply each word's count (**term frequency, TF**) by how *rare* the
word is across documents (**inverse document frequency, IDF**). Words that are everywhere get crushed
toward zero; the distinctive, telling words stand out.
```
IDF(word) = ln( N_docs / N_docs_containing_word )       (+ smoothing in practice)
TF-IDF    = TF · IDF
```
**Worked example by hand** (N=3 docs):
```
"the" is in all 3 docs:  IDF = ln(3/3) = ln(1) = 0      -> TF-IDF 0 everywhere (correctly ignored)
"cat" is in 2 docs:      IDF = ln(3/2) = 0.405
"dog" is in 1 doc:       IDF = ln(3/1) = 1.099           -> rarest, highest weight
```
So `"dog"` (unique to doc1) becomes its strongest signal, while `"the"` vanishes.
""")

code(r'''
import numpy as np
corpus = ["the cat sat", "the dog sat", "the cat ran"]
vocab = ["cat","dog","ran","sat","the"]; stoi={w:i for i,w in enumerate(vocab)}
N = len(corpus)
TF = np.zeros((3,5))
for d,doc in enumerate(corpus):
    for w in doc.split(): TF[d, stoi[w]] += 1

df = (TF > 0).sum(axis=0)                       # docs containing each word
idf = np.log(N / df)
print("word :", vocab)
print("df   :", df.astype(int), " (docs containing word)")
print("idf  :", np.round(idf, 3), " (hand: cat .405, dog 1.099, ran 1.099, sat .405, the 0)")
tfidf = TF * idf
print("\nTF-IDF matrix (note 'the' column is all 0):\n", np.round(tfidf, 3))
''')

code(r'''
# the same with scikit-learn (uses smoothed IDF, so numbers differ slightly but the idea is identical)
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ["the cat sat", "the dog sat", "the cat ran"]
v = TfidfVectorizer()
M = v.fit_transform(corpus)
print("sklearn vocab:", v.get_feature_names_out())
print("sklearn TF-IDF (rows=docs):\n", M.toarray().round(3))
print("'the' still gets the lowest weight — common words are de-emphasised.")
''')

md(r"""
## B4 — N-gram language model: predicting the next word by counting

Here is the moment the mind first tries to *predict* — the great-grandparent of GPT. Think of how you
guess what comes after "once upon a…": you've heard "time" follow it a thousand times. An **n-gram
model** does literally that — it predicts the next word from the previous `n−1` words by **counting**
how often each continuation appeared in training text:
```
P(next = w | previous words) = count(previous words, w) / count(previous words)
```
A **bigram** model (n=2) uses just the last word.

**Worked example by hand.** Corpus `"the cat sat" / "the dog sat" / "the cat ran"`. After `"the"`,
what comes next?
```
"the cat" occurs 2 times, "the dog" occurs 1 time   -> total 3 continuations of "the"
P(cat | the) = 2/3 = 0.667
P(dog | the) = 1/3 = 0.333
```
This is *literally* next-word prediction (Notebook 00) — GPT just replaces counting with a learned
neural network that generalises to word combinations never seen in training.
""")

code(r'''
from collections import defaultdict, Counter
corpus = ["the cat sat", "the dog sat", "the cat ran"]
bigram = defaultdict(Counter)
for doc in corpus:
    w = doc.split()
    for a, b in zip(w[:-1], w[1:]):
        bigram[a][b] += 1

def predict(word):
    counts = bigram[word]; total = sum(counts.values())
    return {w: round(c/total, 3) for w, c in counts.items()}

print("P(next | 'the') :", predict("the"), " (hand cat 0.667, dog 0.333)")
print("P(next | 'cat') :", predict("cat"))
print("P(next | 'sat') :", predict("sat"), " ('sat' ends sentences -> no continuation seen)")
''')

md(r"""
## B5 — Why n-grams fail (and neural models win)

So the mind can read and even guess — but it's still only *sounding out words*, never understanding
them. Here's where the clumsy first reading lessons hit their ceiling. N-gram counting has three
crippling limits, and curing them is the whole arc of Parts 2–4:
1. **No generalisation.** If `"the cat"` was never seen, `P(? | the cat) = 0/0` — undefined. The
   model only knows exact phrases it counted (sparsity explodes as `n` grows).
2. **Short memory.** A 3-gram only sees 2 words back — it can't use long-range context (the same
   problem RNNs had, P16).
3. **No notion of similarity.** `"cat"` and `"kitten"` are unrelated integers; the model can't
   transfer what it learned about one to the other.

Neural language models fix all three by representing words as **embeddings** (vectors where similar
words are close — P18) and using **attention** to reach far back (Notebook 00 / `notebooks/04`).
That's the whole arc of Part 3 → Part 4.
""")

code(r'''
from collections import defaultdict, Counter
corpus = ["the cat sat", "the dog sat", "the cat ran"]
bigram = defaultdict(Counter)
for doc in corpus:
    w = doc.split()
    for a,b in zip(w[:-1], w[1:]): bigram[a][b]+=1
# unseen context -> the model has nothing
print("P(next | 'kitten') :", dict(bigram["kitten"]), " <- empty! word never seen")
print("P(next | 'ran')    :", dict(bigram["ran"]), " <- empty! 'ran' only appeared at sentence end")
print("\nzero generalisation: this is exactly what embeddings + neural nets overcome.")
''')

md(r"""
## Recap — classical text representations

| Method | Idea | Limitation |
|--------|------|------------|
| Tokenization | text → token ids | needs a fixed vocabulary |
| Bag-of-Words | document → word counts | ignores order |
| TF-IDF | weight by rarity | still order-free, no meaning |
| N-gram LM | next word by counting | no generalisation, short memory |

These map text to numbers but treat words as unrelated symbols. **Embeddings (P18)** give words
*meaning*; **attention (Part 4)** gives *long-range, order-aware* context.

## Common mistakes
1. **Forgetting to lowercase / normalise** → `"The"` and `"the"` become different tokens, bloating the vocab.
2. **Using raw counts when common words dominate.** TF-IDF or stopword removal fixes it.
3. **Thinking BoW captures meaning** — `"dog bites man"` and `"man bites dog"` have identical BoW.
4. **Growing n in n-grams to fix memory** → combinatorial explosion and mostly-zero counts.
5. **Treating token ids as numbers with magnitude** — id 5 isn't "more" than id 2; they're just labels (embeddings fix this).

## Exercises (do them in new code cells)
1. Add `"the dog ran"` to the corpus. Recompute `P(next | 'the')` and `P(next | 'dog')` by hand, then verify.
2. Build the BoW vector for the new sentence `"the cat sat the cat"`. Why is the `cat` count 2?
3. Compute the IDF of `"sat"` by hand for the 3-doc corpus. Why is it lower than `"dog"`'s?
4. Make a *trigram* model (condition on the last 2 words). What does it predict after `"the cat"`?
5. Explain, using the n-gram limits, why a GPT can complete `"the kitten ___"` sensibly but a bigram model cannot.

---

**The mind can read words, but it can't yet *mean* them.** To it, `"cat"` and `"kitten"` are still
just two unrelated id numbers — as alien to each other as `0` and `57`. That's the wall counting can
never climb. Next, in [P18 — Word Embeddings from scratch](P18_Word_Embeddings.ipynb), something
beautiful happens: **words become places** in space, and **meaning becomes a direction** — so close
words sit close, and `king − man + woman ≈ queen` falls out as plain geometry. This is the
representation every transformer is built on.
""")

build("P17_Classical_NLP", execute=False)
