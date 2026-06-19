// Curriculum metadata. The playbook is told as ONE story — "Build a Brain": from a single
// number to a thinking machine. `story` is each chapter's narrative hook; `built` marks lessons
// with a rich interactive page.

export type Lesson = {
  id: string;
  code: string;
  title: string;
  desc: string;
  story: string;
  part: string;
  built?: boolean;
};
export type Part = { name: string; icon: string; blurb: string; lessons: Lesson[] };

export const PARTS: Part[] = [
  {
    name: "Part 0 · Foundations", icon: "🔢", blurb: "The mind is born — as nothing but numbers. Teach it the alphabet of thought.",
    lessons: [
      { id: "p00", code: "P00", title: "Python & NumPy", desc: "Arrays, shapes, broadcasting, vectorisation.", story: "Day one. The mind opens its eyes and the whole world is just numbers in motion. Meet them.", part: "Foundations" },
      { id: "p01-dot-product", code: "P01a", title: "The Dot Product", desc: "Drag two vectors; alignment becomes one number.", story: "The mind's first sense: it learns to feel how much two things agree — and answer with a single number.", part: "Foundations", built: true },
      { id: "p01-matrix", code: "P01b", title: "Matrices Transform Space", desc: "Bend the unit square: scale, shear, rotate.", story: "Now it learns to bend space itself. A grid of numbers becomes its very first thought.", part: "Foundations", built: true },
      { id: "p02-gradient-descent", code: "P02", title: "Gradient Descent", desc: "Roll a ball down a 3-D loss valley.", story: "We give the newborn mind a will: a goal, and the urge to stumble downhill toward it. This is how it will learn everything.", part: "Foundations", built: true },
      { id: "p03", code: "P03", title: "Probability & Cross-Entropy", desc: "Drag a distribution; watch its entropy.", story: "The world is uncertain. The mind learns to weigh its hunches — and we learn how to grade its guesses.", part: "Foundations", built: true },
      { id: "p04", code: "P04", title: "Data & Visualisation", desc: "Load, clean, normalise, and see data.", story: "Before it can grow, the mind must eat. Its only food is data — so first we learn to cook.", part: "Foundations" },
    ],
  },
  {
    name: "Part 1 · Classical ML", icon: "📐", blurb: "Before neurons — the mind's first reflexes, and its first hard lesson in humility.",
    lessons: [
      { id: "p05", code: "P05", title: "The ML Mindset", desc: "Slide a polynomial's degree; watch overfitting.", story: "The mind's first temptation: to cheat by memorising instead of truly understanding. We catch it in the act.", part: "Classical ML", built: true },
      { id: "p06", code: "P06", title: "Linear Regression", desc: "Fit the best line with gradient descent.", story: "Its first real skill: draw the single line that best explains a messy world — using that same downhill heartbeat.", part: "Classical ML" },
      { id: "p07", code: "P07", title: "Logistic Regression", desc: "Train a live decision boundary.", story: "It learns to take a side. The line becomes a decision: yes or no, spam or not, cat or dog.", part: "Classical ML", built: true },
      { id: "p08", code: "P08", title: "Evaluation & Regularization", desc: "Precision/recall, ROC, bias–variance.", story: "How do we know the mind is actually right — and not just lucky? We learn to interrogate it honestly.", part: "Classical ML" },
      { id: "p09", code: "P09", title: "Core Algorithms Tour", desc: "kNN, trees, forests, SVM, k-means, PCA.", story: "A whole starter kit of instincts — before we dare to grow real neurons.", part: "Classical ML" },
    ],
  },
  {
    name: "Part 2 · Deep Learning", icon: "🧠", blurb: "A real brain takes shape — neurons, the flash of learning, then eyes and memory.",
    lessons: [
      { id: "p10", code: "P10", title: "Neural Networks", desc: "Add hidden neurons until a net cracks XOR.", story: "We wire many tiny decisions together — and, suddenly, a brain appears that can do what no single neuron could.", part: "Deep Learning", built: true },
      { id: "p11", code: "P11", title: "Backpropagation", desc: "Step a computation graph forward & backward.", story: "The flash of learning itself: the moment the network feels its own mistake and fixes it. This is the spark.", part: "Deep Learning", built: true },
      { id: "p12", code: "P12", title: "Optimizers", desc: "Race SGD, Momentum, RMSProp & Adam.", story: "Now we teach it to learn faster — to sprint downhill without tripping over its own feet.", part: "Deep Learning", built: true },
      { id: "p13", code: "P13", title: "Training Deep Nets", desc: "Init, normalization, dropout, residuals.", story: "As the brain grows deep, it threatens to fall apart. These are the tricks that hold a deep mind together.", part: "Deep Learning" },
      { id: "playground", code: "P14", title: "Neural Network Playground", desc: "Train an MLP live; watch the decision boundary.", story: "Step into the lab. Build a living brain with your own hands and watch it learn, right here in your browser.", part: "Deep Learning", built: true },
      { id: "p15", code: "P15", title: "Convolutional Networks", desc: "Slide a kernel; watch a feature map form.", story: "We give the mind eyes. First it sees edges, then shapes, then — things.", part: "Deep Learning", built: true },
      { id: "p16", code: "P16", title: "Recurrent Networks", desc: "Memory, and why attention was born.", story: "We give it memory, so it can read a sentence one word at a time — and discover exactly why that memory wasn't enough.", part: "Deep Learning" },
    ],
  },
  {
    name: "Part 3 · Language", icon: "💬", blurb: "The mind learns to read — and, for the first time, to mean.",
    lessons: [
      { id: "p17", code: "P17", title: "Classical NLP", desc: "Tokenization, BoW, TF-IDF, n-grams.", story: "Its first, clumsy reading lessons: it counts words like a tourist with a phrasebook — and misses the meaning entirely.", part: "NLP" },
      { id: "p18", code: "P18", title: "Word Embeddings", desc: "Words in 3-D space; king − man + woman.", story: "The breakthrough: words become places, and meaning becomes a direction you can point to. King − man + woman = queen.", part: "NLP", built: true },
      { id: "p19", code: "P19", title: "Seq2Seq & Attention", desc: "The bottleneck that birthed attention.", story: "Reading through a single keyhole, the mind chokes on long sentences — until it learns to pay attention to what matters.", part: "NLP" },
    ],
  },
  {
    name: "Part 4 · Transformers", icon: "⚡", blurb: "Every piece snaps together into the mind's true form.",
    lessons: [
      { id: "p20", code: "P20", title: "Attention → Transformer", desc: "Assemble a block; bridge into TinyGPT.", story: "The assembly scene: every part you've built clicks into place as the architecture behind modern AI — and hands off to TinyGPT.", part: "Transformers" },
    ],
  },
  {
    name: "Part 5 · LLM Engineering", icon: "🚀", blurb: "Raising a real mind — feed it the world, then teach it manners, judgement, and tools.",
    lessons: [
      { id: "p21", code: "P21", title: "Pretraining & Scaling", desc: "Self-supervised objective, scaling laws.", story: "We feed it the whole internet and ask one question, a trillion times: what word comes next? The same loop — a billion-fold bigger.", part: "LLM Engineering" },
      { id: "p22", code: "P22", title: "Supervised Fine-Tuning", desc: "From predictor to instruction-follower.", story: "The raw mind is a brilliant mumbler. We teach it manners: how to actually answer the question it was asked.", part: "LLM Engineering" },
      { id: "p23", code: "P23", title: "LoRA & Quantization", desc: "Fine-tune giant models cheaply.", story: "How do you reshape a giant mind on a single laptop? You don't move the mountain — you add a few clever footpaths.", part: "LLM Engineering" },
      { id: "p24", code: "P24", title: "RLHF: Reward Models & DPO", desc: "Tune the preference loss.", story: "We teach it taste — not just a right answer, but the better of two good ones. We teach it what people actually prefer.", part: "LLM Engineering", built: true },
      { id: "p25", code: "P25", title: "RAG, Tools & Agents", desc: "External knowledge and actions.", story: "We hand it a library so it stops making things up — and then tools, so it can act, not just talk.", part: "LLM Engineering" },
      { id: "p26", code: "P26", title: "Prompting & Decoding", desc: "Temperature, top-k, top-p — then sample.", story: "Same mind, many moods. We learn the dials that turn its voice from careful and factual to wild and creative.", part: "LLM Engineering", built: true },
      { id: "p27", code: "P27", title: "Evaluation & Safety", desc: "Perplexity, benchmarks, guardrails.", story: "Is it truly smart, or just confident? And can we trust it? We learn to measure the mind — and to keep it safe.", part: "LLM Engineering" },
    ],
  },
  {
    name: "Part 6 · Deployment", icon: "🛰️", blurb: "The mind leaves home — out into the world, fast and accountable.",
    lessons: [
      { id: "p28", code: "P28", title: "Serving & Inference", desc: "KV-cache, batching, quantized inference.", story: "A mind no one can reach isn't much use. We make it fast and cheap enough to answer millions at once.", part: "Deployment" },
      { id: "p29", code: "P29", title: "MLOps", desc: "Tracking, versioning, monitoring drift.", story: "The story doesn't end at launch. We keep the mind alive, healthy, and honest as the world keeps changing around it.", part: "Deployment" },
    ],
  },
];

export const ALL_LESSONS: Lesson[] = PARTS.flatMap((p) => p.lessons);
export const getLesson = (id: string) => ALL_LESSONS.find((l) => l.id === id);
export const lessonIndex = (id: string) => ALL_LESSONS.findIndex((l) => l.id === id);
export const partOf = (id: string) => PARTS.find((p) => p.lessons.some((l) => l.id === id));
export const TOTAL = ALL_LESSONS.length;
