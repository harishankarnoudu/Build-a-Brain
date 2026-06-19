import type { ReactNode } from "react";
import { M, MathBlock } from "../components/Math";
import { ShowMath } from "../components/Details";
import { Scene, RealWorld } from "../components/Story";

// Prose chapters (no interactive yet) of the one story — "Raising a Mind".
// Each opens on a real-world scene, earns its idea in plain words + a hand example,
// keeps heavy math behind ShowMath, and ends looking forward to the next stage.
// The matching playbook/ notebook remains the deep, runnable version.
export const ARTICLES: Record<string, ReactNode> = {
  p00: (
    <>
      <Scene icon="👶">
        <p>It's day one. The mind we're about to raise opens its eyes — and sees no colours, no words,
          no shapes. It sees <strong>numbers</strong>. A photo is a grid of numbers. A sentence is a list of
          numbers. A sound is numbers over time. Every thought it will ever have is numbers being
          multiplied and added, fast.</p>
      </Scene>
      <p>So before "machine learning" can mean anything, we need the thing that holds and moves numbers
        in bulk: <strong>NumPy</strong>. PyTorch (which we use later) copies its style exactly, so learning
        NumPy is learning the language the whole field is written in.</p>

      <h2>Numbers travel in packs: the array</h2>
      <p>A NumPy array is a grid of numbers, and math happens to the <em>whole grid at once</em>. The single
        most important thing about an array is its <strong>shape</strong> — read it obsessively, because nearly
        every bug you'll ever hit is a shape that didn't line up.</p>
      <div className="hand">{`a = [1, 2, 3]          shape (3,)      a vector — 3 numbers in a row
M = [[1,2,3],[4,5,6]]  shape (2, 3)    2 rows, 3 columns
a * 2  -> [2, 4, 6]    (every element doubles — not the list repeated)`}</div>

      <h2>Do it to everyone at once: vectorisation &amp; broadcasting</h2>
      <p>Here's the mental shift. You do <em>not</em> write a loop that touches one number at a time — you
        speak to the whole array in one breath. That's <strong>vectorisation</strong>: shorter to write and
        ~100× faster, because the work happens in optimised C, not Python.</p>
      <p><strong>Broadcasting</strong> is the quiet rule that lets arrays of different shapes still combine:
        the smaller one is "stretched" to fit. It's not a footnote — it's <em>exactly</em> how a single bias
        gets added to every row a layer produces.</p>
      <div className="hand">{`M + [10,20,30]   adds that row to EVERY row of M (it's stretched down)
M.sum(axis=0) -> per-column totals     M.sum(axis=1) -> per-row totals`}</div>

      <RealWorld>When your phone brightens a dark photo, it isn't looping pixel-by-pixel in slow motion —
        it adds a number to a whole array of brightness values in one vectorised stroke. Same idea, same speed trick.</RealWorld>

      <div className="callout">Master three things — <strong>shapes</strong>, <strong>vectorisation</strong>, and
        <strong> broadcasting</strong> — and the entire rest of this journey is just these operations, stacked
        deep. Next, the mind gets its first real <em>sense</em>: the <strong>dot product</strong>.</div>
    </>
  ),

  p04: (
    <>
      <Scene icon="🍳">
        <p>You wouldn't feed a newborn straight from a can without checking it first. The mind is the same:
          it grows into whatever you feed it. Feed it messy, lopsided, half-missing data and it grows into a
          confused adult. So before any learning happens, we become the cook — we shop, inspect, clean, and
          plate the <strong>data</strong>.</p>
      </Scene>

      <h2>Look before you cook: the DataFrame</h2>
      <p>A pandas <strong>DataFrame</strong> is just a table with named columns — a spreadsheet you can
        command in code. The discipline that separates good practitioners from frustrated ones is simple:
        <strong> always look at the data first</strong>. <code>.head()</code> to see real rows,
        <code> .describe()</code> for the ranges, <code>.shape</code> for the size. Never train on food you
        haven't tasted.</p>

      <h2>Fix what's broken, then put everyone on the same scale</h2>
      <p>Real data has holes (missing values) and columns measured in wildly different units — a person's
        age (0–100) sitting next to their income (0–1,000,000). To the mind, the bigger numbers <em>shout</em>
        and drown out the rest. So we fill the holes (often with the column's mean or median) and
        <strong> standardise</strong> every column to the same quiet scale — mean 0, spread 1. We subtract the
        column's average <M>{"\\mu"}</M> (so it's centred on 0) and divide by its standard deviation
        <M>{" \\sigma"}</M> (the typical distance from the average, so the spread becomes 1):</p>
      <MathBlock>{"z = \\frac{x - \\mu}{\\sigma}"}</MathBlock>
      <p>Watch it on three ages, by hand. The "+1.22" means "1.22 standard deviations above average" — a unit
        that means the same thing for age, income, or anything else:</p>
      <div className="hand">{`ages = [20, 30, 40]
mean  μ = (20+30+40)/3 = 30
var     = ((20−30)² + (30−30)² + (40−30)²)/3 = (100+0+100)/3 = 66.7
std   σ = √66.7 ≈ 8.16
z(20) = (20−30)/8.16 = −1.22     z(30) = 0     z(40) = +1.22`}</div>
      <p>Now "30 years old" and "$60,000" both become small numbers around 0, so neither shouts over the other,
        and the mind can weigh them fairly.</p>

      <h2>Plating up: features and labels</h2>
      <p>The last step splits the table into the <strong>features <M>{"X"}</M></strong> (what the mind sees)
        and the <strong>label <M>{"y"}</M></strong> (what we want it to predict). That <M>{"(X, y)"}</M> pair
        is the meal every model in this story eats.</p>

      <div className="callout">One rule that saves careers: compute your cleaning statistics (means, scales)
        on the <em>training</em> data only, then reuse them on validation/test. Peek at the test set and your
        scores become a comforting lie. Now the kitchen's ready — let's teach the mind its first instinct.</div>
    </>
  ),

  p06: (
    <>
      <Scene icon="📈">
        <p>A landlord wants to guess an apartment's rent from its size. More square metres, more rent —
          roughly a straight line, but the dots are scattered. Drawing the <em>best</em> line through that
          scatter is the mind's very first skill, and it's called <strong>linear regression</strong>.</p>
      </Scene>
      <p>It's the "hello world" of machine learning — and, secretly, it runs the <em>exact</em> same training
        loop as a giant neural network. Learn it here and you've learned the heartbeat of everything that follows.</p>

      <h2>A line, and a way to score it</h2>
      <p>The line is <M>{"\\hat{y} = w x + b"}</M>: weight <M>{"w"}</M> is the slope, bias <M>{"b"}</M> is where
        it starts. To improve it, we need to measure how wrong it is. We use <strong>mean squared error</strong> —
        average the squared gaps between guess and truth:</p>
      <MathBlock>{"\\text{MSE} = \\frac{1}{n}\\sum_i (\\hat{y}_i - y_i)^2"}</MathBlock>
      <p>Squaring makes every miss positive and punishes big mistakes far harder than small ones. Lower MSE =
        better line. That's the whole scorecard.</p>

      <h2>The heartbeat: predict → measure → step downhill</h2>
      <p>Here it is — the rhythm you'll meet again in neural nets and in GPT itself. Differentiate the error to
        learn which way is "downhill," then nudge the line that way, over and over:</p>
      <ShowMath>
        <MathBlock>{"\\frac{\\partial \\text{MSE}}{\\partial w} = \\frac{2}{n}\\sum_i (\\hat{y}_i - y_i)\\,x_i, \\qquad w \\leftarrow w - \\eta\\,\\frac{\\partial \\text{MSE}}{\\partial w}"}</MathBlock>
      </ShowMath>
      <p>Notice the shape of that gradient: <strong>error × input</strong>. That same form shows up in logistic
        regression, in a neuron, in a transformer. Let's take <em>one</em> step by hand on a single point so the
        loop stops being abstract — a flat (size = 2, rent = 6) line that starts at zero:</p>
      <div className="hand">{`start: w = 0, b = 0   point: x = 2, y = 6   step size η = 0.01
predict   ŷ = w·x + b = 0·2 + 0 = 0
error     (ŷ − y) = 0 − 6 = −6          ← we guessed 6 too low
gradient  ∂/∂w = 2·(ŷ−y)·x = 2·(−6)·2 = −24    (error × input)
          ∂/∂b = 2·(ŷ−y)   = 2·(−6)   = −12
step      w ← 0 − 0.01·(−24) = +0.24      b ← 0 − 0.01·(−12) = +0.12
new ŷ = 0.24·2 + 0.12 = 0.60   ← was 0, now closer to the true 6`}</div>
      <p>One nudge moved the guess from 0 toward 6. Repeat it thousands of times across all the points and the
        line settles onto the data. That loop — <em>predict → measure how wrong → step downhill</em> — is the
        engine of all of deep learning.</p>

      <RealWorld>Every "estimated delivery: 25 min", "your home is worth ~₹X", and trend line in a spreadsheet
        is this same line-fitting, scaled to more inputs.</RealWorld>

      <div className="callout">With many features the model is just one matrix multiply,
        <M>{"\\;\\hat{y} = Xw + b"}</M> — precisely what a <code>Linear</code> layer computes. Next: turn this line
        into a <em>decision</em>.</div>
    </>
  ),

  p08: (
    <>
      <Scene icon="🩺">
        <p>A cancer screening test brags that it's <strong>99% accurate</strong>. Sounds brilliant — until you
          realise that if only 1 in 100 patients is actually sick, a test that just says "healthy" to
          <em> everyone</em> also scores 99%. It catches zero cancers and still looks great on paper. Accuracy
          lied. The mind needs us to grade it more honestly than that.</p>
      </Scene>

      <h2>Beyond accuracy: what kind of wrong?</h2>
      <p>Not all mistakes are equal. The <strong>confusion matrix</strong> sorts predictions into true/false
        positives and negatives, and from it come the two numbers that actually matter:</p>
      <MathBlock>{"\\text{precision} = \\frac{TP}{TP+FP}, \\quad \\text{recall} = \\frac{TP}{TP+FN}, \\quad F_1 = \\frac{2\\,PR}{P+R}"}</MathBlock>
      <p>(TP = correct "sick" calls, FP = false alarms, FN = missed cases.) Let's grade two models on those 100
        patients by hand — and watch accuracy and recall completely disagree:</p>
      <div className="hand">{`100 patients, only 1 truly sick.

Model A — "everyone is healthy":
  TP=0  FP=0  FN=1  TN=99
  accuracy = 99/100 = 99%      ← looks amazing
  recall   = 0/(0+1) = 0%      ← caught the cancer? NO. useless.

Model B — flags 5 people, catching the 1 real case + 4 false alarms:
  TP=1  FP=4  FN=0  TN=95
  recall    = 1/(1+0) = 100%   ← caught it!
  precision = 1/(1+4) = 20%    ← 4 of 5 alarms were wrong`}</div>
      <p>Same data, opposite verdicts. A cancer screen lives or dies on <strong>recall</strong> (Model B wins —
        catch every real case, tolerate false alarms). A spam filter prizes <strong>precision</strong> — never
        trash a real email, even if a little spam slips through. <strong>F₁</strong> blends the two when you care
        about both, and <strong>ROC/AUC</strong> sums up how well the model <em>ranks</em> cases across every
        possible threshold.</p>

      <h2>Teaching humility: regularization &amp; the bias–variance tug-of-war</h2>
      <p>Remember the mind's temptation to memorise? We fight it by <strong>penalising large weights</strong> so
        the model can't contort itself to fit noise. <strong>L2</strong> (ridge) shrinks weights smoothly;
        <strong> L1</strong> (lasso) drives some to exactly zero. Every model's error splits into
        <strong> bias</strong> (too simple, misses the pattern) and <strong>variance</strong> (too jumpy, chases
        noise) — and good learning is finding the balance that does best on data it has <em>never seen</em>.</p>

      <div className="callout">In deep learning this same L2 penalty is renamed <strong>weight decay</strong> and
        is switched on by default in modern optimizers. The lesson the mind learns here — <em>don't trust a single
        flattering number</em> — never stops mattering.</div>
    </>
  ),

  p09: (
    <>
      <Scene icon="🧰">
        <p>Before we grow real neurons, the mind deserves a starter kit — a drawer of simple, reliable
          instincts. Honestly, on an everyday spreadsheet these still <em>beat</em> deep learning, with a
          fraction of the fuss. And each one quietly teaches an idea that comes roaring back later.</p>
      </Scene>

      <h2>The line-up</h2>
      <ul>
        <li><strong>k-Nearest Neighbours</strong> — "you are who you sit with." Classify a point by the majority
          vote of its closest neighbours. No training at all, just memory and distance.</li>
        <li><strong>Decision Trees</strong> — a flowchart the mind <em>learns</em>: a stack of yes/no questions.
          Wonderfully easy to read out loud.</li>
        <li><strong>Random Forests</strong> — ask a whole crowd of trees and average them (an <em>ensemble</em>).
          Accurate, robust, hard to fool.</li>
        <li><strong>SVMs</strong> — draw the boundary with the widest possible no-man's-land around it; kernels
          let that boundary curve.</li>
        <li><strong>Naive Bayes</strong> — Bayes' rule plus a cheeky "assume everything's independent." Crude,
          fast, and weirdly great at text.</li>
        <li><strong>k-Means</strong> — no labels at all: "assign each point to the nearest cluster centre, move
          the centres, repeat." Self-organising order.</li>
        <li><strong>PCA</strong> — keep only the directions where the data actually varies, to compress and to
          <em> see</em> high-dimensional data.</li>
      </ul>

      <p>Let's watch the simplest one, <strong>kNN</strong>, decide by hand — it's nothing but "measure distance,
        take a vote":</p>
      <div className="hand">{`Known fruits (weight g, label):  apple@150, apple@160, lemon@120, lemon@110
New fruit weighs 155g. Use k=3 nearest:
  distances: |155−150|=5(apple), |155−160|=5(apple), |155−120|=35(lemon), |155−110|=45(lemon)
  3 nearest = apple(5), apple(5), lemon(35)  → vote 2 apple : 1 lemon  → APPLE`}</div>
      <p>No training, no formula learned — just stored examples and a distance. That's the whole algorithm, and it
        already hints at the idea this whole journey rests on: <em>similar things sit close together</em>.</p>

      <RealWorld>Your bank's fraud check, the "customers also bought" panel, and the credit-score cutoff are
        often one of these — not a neural net. Simple instincts, doing real work.</RealWorld>

      <div className="callout">Rule of thumb: reach for a <strong>random forest</strong> first on a small table —
        it'll often win. Save the neural networks for images, sound, and language, where the patterns are too
        tangled for any flowchart. Speaking of which — it's time to grow some neurons.</div>
    </>
  ),

  p13: (
    <>
      <Scene icon="🏗️">
        <p>The brain wants to grow <em>deep</em> — layer upon layer. But naively stacking layers is like
          building a tower with no foundation: the learning signal either fades to a whisper before it reaches
          the bottom (<strong>vanishing gradients</strong>) or blows up into nonsense (<strong>exploding
          gradients</strong>), and training simply stalls. Four engineering tricks are what finally let deep
          learning <em>work</em>.</p>
      </Scene>

      <h2>The four load-bearing tricks</h2>
      <ul>
        <li><strong>Initialization</strong> (He / Xavier) — start the weights at just the right scale so the
          signal neither shrinks nor swells as it passes through each layer.</li>
        <li><strong>Normalization</strong> — <strong>LayerNorm</strong> re-centres each token's features to mean
          0, spread 1, keeping activations well-behaved. It's in every single transformer block.</li>
        <li><strong>Dropout</strong> — randomly switch off some neurons during training so the network can't lean
          on any one of them. A cheap, powerful cure for overfitting.</li>
        <li><strong>Residual connections</strong> — add a layer's input back to its output, giving the learning
          signal a <em>highway</em> straight down through depth instead of a hundred flights of stairs.</li>
      </ul>
      <ShowMath>
        <MathBlock>{"\\text{LayerNorm: } \\hat{x} = \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}}, \\qquad \\text{residual: } \\text{out} = x + f(x)"}</MathBlock>
      </ShowMath>

      <p>LayerNorm is just the standardise trick from P04, applied to one token's features. By hand on a vector
        that's drifted large: <code>x = [2, 4, 6]</code> → mean <code>μ = 4</code>, std <code>σ ≈ 1.63</code> →
        <code> x̂ = [−1.22, 0, +1.22]</code>. Tidied back to mean 0, spread 1, every layer — so activations never
        snowball or starve.</p>
      <p>And the reason a <strong>residual</strong> <M>{"\\text{out} = x + f(x)"}</M> rescues the gradient is one
        line of calculus: its slope is <M>{"1 + f'(x)"}</M>. That <strong>+1</strong> means the learning signal
        always has a clear path straight through — even when the layer's own <M>{"f'(x)"}</M> is tiny, the
        gradient can't vanish to nothing.</p>

      <div className="callout">These four aren't trivia — they're <em>why</em> a transformer block
        (<code>x → LayerNorm → attention → +x → LayerNorm → FFN → +x</code>) trains at all. You'll see exactly
        this skeleton in Part 4. Now, into the lab.</div>
    </>
  ),

  p16: (
    <>
      <Scene icon="📖">
        <p>Reading a sentence is nothing like seeing a photo. A photo arrives all at once; a sentence arrives
          one word at a time, and the meaning of "<em>bank</em>" depends on what came ten words ago. To read,
          the mind needs something new: <strong>memory</strong>. That's the <strong>Recurrent Neural
          Network</strong> — it reads one word, updates a running memory, and carries it forward.</p>
      </Scene>

      <h2>The recurrence: a memory that walks forward</h2>
      <MathBlock>{"h_t = \\tanh(W_x x_t + W_h h_{t-1} + b)"}</MathBlock>
      <p>The same weights fire at every step. The hidden state <M>{"h_t"}</M> is the mind's running summary of
        everything it's read so far; the final state is its understanding of the whole sentence. Two steps by
        hand, with tiny weights, show the memory building up — each new <M>{"h"}</M> mixes the new word
        <em> and</em> the old memory:</p>
      <div className="hand">{`W_x = 0.5, W_h = 0.5, b = 0,  each word x = 2
step 1 (h₀=0):  h₁ = tanh(0.5·2 + 0.5·0) = tanh(1.0)  ≈ 0.76
step 2:         h₂ = tanh(0.5·2 + 0.5·0.76) = tanh(1.38) ≈ 0.88   ← carries h₁ forward`}</div>

      <h2>Why its memory fades — and the wall it hit</h2>
      <p>There's a catch. Each step multiplies the old memory's influence by a factor below 1, so it shrinks
        geometrically — the same <strong>vanishing-gradient</strong> problem, now stretched across <em>time</em>.
        If each step keeps just half: <code>0.5¹⁰ ≈ 0.001</code> — after ten words the first word is essentially
        gone. So a plain RNN forgets the start of a long paragraph by the time it reaches the end. <strong>LSTMs</strong> bolted on a
        gated memory cell — a near-direct path for information to survive — and helped a lot. But they were still
        stubbornly <em>sequential</em>: they had to read word-by-word, couldn't be parallelised, and squeezed
        everything through one small memory vector.</p>

      <div className="callout"><strong>Attention</strong> shatters both limits at once: every position can look
        <em> directly</em> at every other position, all in parallel, with no fading over distance. That single
        idea, scaled up, becomes the transformer. The mind is about to learn to <em>focus</em>.</div>
    </>
  ),

  p17: (
    <>
      <Scene icon="🗂️">
        <p>Imagine reading a foreign language armed only with a frequency list — which words appear, and how
          often. You'd catch the gist of a news article ("money… market… rises") and miss every joke, every
          negation, every shade of meaning. That's how machines read <em>before</em> they learned real meaning:
          by counting. Crude — but these methods are still useful baselines, and one of them predicts the next
          word in <em>exactly</em> the way GPT does, just by tallying instead of learning.</p>
      </Scene>

      <h2>From text to numbers, the old way</h2>
      <ul>
        <li><strong>Tokenization</strong> — chop text into tokens, map each to an integer id.</li>
        <li><strong>Bag-of-Words</strong> — describe a document by its word counts, throwing word <em>order</em>
          away entirely ("dog bites man" = "man bites dog").</li>
        <li><strong>TF-IDF</strong> — quiet down words that appear everywhere ("the"), amplify the words that make
          a document distinctive.</li>
        <li><strong>n-gram language model</strong> — predict the next word from the previous few, purely by
          counting how often that continuation appeared:</li>
      </ul>
      <MathBlock>{"P(\\text{next} \\mid \\text{context}) = \\frac{\\text{count}(\\text{context}, \\text{next})}{\\text{count}(\\text{context})}"}</MathBlock>
      <p>It's pure tallying. Take the toy corpus <em>"the cat sat. the cat ran."</em> and ask: after "cat", what
        comes next?</p>
      <div className="hand">{`count("cat") = 2          (it appears twice)
count("cat","sat") = 1  →  P(sat | cat) = 1/2 = 0.5
count("cat","ran") = 1  →  P(ran | cat) = 1/2 = 0.5`}</div>
      <p>That's it — that <em>is</em> the language model. GPT answers the very same question ("what's next?"), but
        instead of counting it <em>learns</em> the probabilities, which is why it can handle phrases it has never
        literally seen.</p>

      <RealWorld>The earliest phone keyboards and search autocompletes ran on n-grams — which is why they felt
        so robotic the moment you wrote anything original.</RealWorld>

      <div className="callout">The fatal flaw: counting can't <em>generalise</em>. An unseen phrase gets
        probability zero, and memory is tiny. The fix is two leaps — give words real <strong>meaning</strong>
        (embeddings, next) and give them long-range <strong>context</strong> (attention). That's the road to
        transformers.</div>
    </>
  ),

  p19: (
    <>
      <Scene icon="🌐">
        <p>Picture a translator who must listen to an entire German sentence, memorise it perfectly, and only
          <em> then</em> open their mouth to speak English — never glancing back. For a short sentence, fine.
          For a long, winding one, they crack. That was the state of machine translation: an
          encoder–decoder ("seq2seq") that crammed the whole input into one fixed summary vector. Then someone
          asked: <em>what if the translator could glance back at any word, whenever they needed it?</em> That
          question gave us <strong>attention</strong>.</p>
      </Scene>

      <h2>The bottleneck</h2>
      <p>An encoder RNN squeezes the entire input sentence into a single vector; the decoder must generate the
        whole translation from that one squeezed summary. For long sentences, one vector simply can't hold
        everything — and quality collapses exactly when you need it most.</p>

      <h2>Attention: stop memorising, start glancing back</h2>
      <p>Instead of one frozen summary, let the decoder build a <strong>fresh, custom blend of all the encoder's
        states</strong> at every step — weighting each input word by how relevant it is right now (a dot product,
        a softmax, then a weighted sum):</p>
      <MathBlock>{"\\text{context} = \\sum_i \\text{softmax}(q \\cdot k_i)\\; v_i"}</MathBlock>
      <p>Notice the hero of Part 0 returns: the <strong>dot product</strong> <M>{"q\\cdot k_i"}</M> is how the
        mind decides "how much should I attend to <em>this</em> word?" Let's run it by hand. The current word asks
        a question <M>{"q"}</M>; each other word offers a key <M>{"k_i"}</M>. Dot products score the match,
        softmax turns those scores into weights that sum to 1, and we blend the values by those weights:</p>
      <div className="hand">{`query q = [1, 0]      key of "river" k₁ = [1, 0]     key of "money" k₂ = [0, 1]
scores:  q·k₁ = 1·1+0·0 = 1        q·k₂ = 1·0+0·1 = 0
softmax([1, 0]) = [ e¹/(e¹+e⁰), e⁰/(e¹+e⁰) ] = [2.72/3.72, 1/3.72] = [0.73, 0.27]
context = 0.73·v(river) + 0.27·v(money)     ← 73% of its attention goes to "river"`}</div>
      <p>The word "bank" just decided, on its own, to listen mostly to "river" — so it will read as a riverbank,
        not a money bank. Direction became a number, and that number became <em>focus</em>.</p>

      <div className="callout">No fading over distance, no single bottleneck, and fully parallel. Now point that
        same trick at a sentence so it can attend to <em>itself</em> — and you've invented <strong>self-attention</strong>,
        the beating heart of the transformer.</div>
    </>
  ),

  p20: (
    <>
      <Scene icon="🧩">
        <p>This is the assembly scene. Every part the mind has grown — the dot product, matrices, the downhill
          heartbeat, probability, neurons, normalization, attention — has been a puzzle piece on the table. Now
          they click together into a single shape: the <strong>transformer</strong>, the architecture behind every
          modern AI you've heard of. There's no new magic here. Only assembly.</p>
      </Scene>

      <h2>The block, built from pieces you already own</h2>
      <div className="hand">{`x ─► LayerNorm ─► self-attention ─► + x   (residual)
  ─► LayerNorm ─► feed-forward    ─► + x   (residual)`}</div>
      <p>Self-attention (Part 3) lets tokens share information; the feed-forward network (the neuron of Part 2)
        thinks about each token on its own; LayerNorm and residual connections (Part 2's survival tricks) keep
        the deep stack trainable. That's the entire block.</p>

      <h2>From one block to a GPT</h2>
      <p>Put token + position embeddings (Part 3) in front, stack <M>{"N"}</M> of these blocks, then cap it with
        the <strong>LM head</strong> — a final linear layer that turns each token's vector into one score per word
        in the vocabulary, which <strong>softmax</strong> (Part 0) converts into next-word probabilities. Train
        the whole thing with cross-entropy (Part 0) and Adam (Part 2) via backprop (Part 2). That one sentence
        names the <em>entire</em> model — and you understand every word in it.</p>

      <div className="callout">This is the doorway. From here the playbook hands off to the
        <strong> TinyGPT notebooks (00–13)</strong>, where you build, train, and probe a real GPT from scratch —
        every part of which you now genuinely understand. The mind has found its true form.</div>
    </>
  ),

  p21: (
    <>
      <Scene icon="🌊">
        <p>Now the mind is fully formed but utterly empty — a brilliant brain that has never read a word. So we
          do something almost absurd: we pour <em>the entire internet</em> through it and ask the same tiny
          question, trillions of times. <strong>What word comes next?</strong> No human grades the answers.
          The text itself is the teacher.</p>
      </Scene>

      <h2>Self-supervised: the text grades itself</h2>
      <p>This is the trick that unlocked modern AI. For any sentence, the "correct answer" for each position is
        simply the word that actually came next — so <em>every piece of text in existence</em> is free training
        data, no labellers required. The loss is the same cross-entropy from Part 0. The unglamorous truth is
        that most of the real work is the <strong>data pipeline</strong>: collect, clean, dedupe, tokenize, pack.</p>

      <h2>Scaling laws: progress you can predict</h2>
      <p>Here's the eerie part. Test loss falls as a smooth <strong>power law</strong> in model size, data, and
        compute: <M>{"L \\approx a\\,N^{-b}"}</M>. Why does that show up as a <em>straight line</em> on a log-log
        plot? Take the log of both sides — <M>{"\\log L = \\log a - b\\,\\log N"}</M> — which is just
        <M>{" y = c - b x"}</M>, a line with slope <M>{"-b"}</M>. So labs fit that line from a few small, cheap
        runs and <em>extrapolate</em> it rightward to predict how a far bigger model will perform, before spending
        millions to build it. The "Chinchilla" lesson: grow parameters and training tokens <em>together</em>
        (~20 tokens per parameter) for the best use of a fixed compute budget.</p>

      <div className="callout">Strip away the scale and pretraining is just next-token prediction with AdamW,
        a warmup, and a cosine decay — <strong>the same training loop from Part 0, a billion-fold bigger</strong>.
        That's the whole secret. Next: it can predict text, but it can't yet <em>answer</em> you.</div>
    </>
  ),

  p22: (
    <>
      <Scene icon="🎓">
        <p>The freshly pretrained mind is a spectacular mumbler. Ask it "What's the capital of France?" and it
          might reply "is a question many students are asked in…" — because it learned to <em>continue</em> text,
          not to <em>answer</em> it. <strong>Supervised fine-tuning</strong> (instruction tuning) is finishing
          school: we teach the brilliant mumbler some manners.</p>
      </Scene>

      <h2>The recipe</h2>
      <p>Keep training the model, but now on curated <strong>(instruction → ideal response)</strong> pairs,
        wrapped in a chat template with role markers ("user", "assistant"). The clever twist: compute the loss
        <strong> only on the response tokens</strong> and mask out the prompt (the special label <code>-100</code>
        tells the loss "ignore this token"). Why? If you scored the prompt tokens too, you'd be rewarding the
        model for predicting the <em>question</em> — but the user supplies the question; we only want it to get
        better at the <em>answer</em>.</p>

      <h2>A gentle nudge, not a rebuild</h2>
      <p>Use a <strong>small learning rate</strong>: the mind is already brilliant from pretraining, and big steps
        would bulldoze that knowledge — a failure called <strong>catastrophic forgetting</strong>, where it
        learns the new format but loses the facts and fluency it had. And quality crushes quantity — a few
        thousand thoughtful, diverse examples beat millions of mediocre ones. You're polishing a diamond, not
        mining a new one.</p>

      <div className="callout">SFT gives you a genuinely helpful assistant. But among several <em>good</em>
        answers, which does a human actually <em>prefer</em>? Teaching that taste is the next stage: RLHF and
        DPO.</div>
    </>
  ),

  p23: (
    <>
      <Scene icon="🗻">
        <p>You want to specialise a 7-billion-parameter mind for your own task — but fully fine-tuning it needs
          tens of gigabytes of GPU memory you don't have. It feels like being told you must physically move a
          mountain to change the path across it. <strong>LoRA</strong> and <strong>quantization</strong> are how
          you reshape a giant brain on a single consumer GPU — by not moving the mountain at all.</p>
      </Scene>

      <h2>LoRA: freeze the giant, add tiny footpaths</h2>
      <p>Leave all those billions of pretrained weights <strong>frozen</strong>. Learn only a small
        <em> correction</em>, cleverly factored into two skinny matrices:</p>
      <MathBlock>{"W_{\\text{new}} = W + \\frac{\\alpha}{r} B A, \\quad B \\in \\mathbb{R}^{d\\times r},\\; A \\in \\mathbb{R}^{r\\times k},\\; r \\ll d,k"}</MathBlock>
      <p>The magic is in the shapes. <M>{"B"}</M> and <M>{"A"}</M> are skinny, so together they hold far fewer
        numbers than <M>{"W"}</M> — yet their product <M>{"BA"}</M> is the same size as <M>{"W"}</M> and can be
        added straight onto it. Count it by hand for one weight of a 7B-class model:</p>
      <div className="hand">{`one weight W: 4096 × 4096      = 16,777,216 numbers   (you'd have to train all of these)
LoRA with r = 8:
  A is 8 × 4096   = 32,768
  B is 4096 × 8   = 32,768
  total trained   = 65,536 numbers  ≈ 0.4% of W`}</div>
      <p>Roughly <strong>0.4%</strong> of the parameters — the footpaths, not the mountain. Same behaviour change,
        a fraction of the memory and time.</p>

      <h2>Quantization &amp; QLoRA: shrink the storage</h2>
      <p>Store the frozen weights in int8 or int4 instead of 32-bit floats — 4–8× less memory, using a simple
        scale factor to map the range. <strong>QLoRA</strong> marries both ideas: a 4-bit frozen base plus small,
        high-precision LoRA adapters on top.</p>

      <div className="callout">The base never receives gradients and barely touches memory; only the featherweight
        adapters learn. That's the trick behind nearly every "fine-tuned on a single GPU" model you've seen.</div>
    </>
  ),

  p25: (
    <>
      <Scene icon="📚">
        <p>Ask the mind about something that happened last week, or a fact buried in your company's private
          wiki, and it does something dangerous: it answers <em>confidently anyway</em>, inventing details. It
          only knows what it read during training, and it can't look anything up. So we give it two human gifts:
          a <strong>library</strong> (RAG) and <strong>hands</strong> (tools and agents).</p>
      </Scene>

      <h2>RAG: let it read before it answers</h2>
      <p>Turn every document into a vector and store it. When a question arrives, turn <em>it</em> into a vector
        too and find the closest documents by <strong>cosine similarity</strong> (Part 3's "meaning is direction"
        again). Paste those top matches into the prompt, so the model answers from <em>sources it can cite</em> —
        slashing hallucination and letting it use fresh or private knowledge it never trained on.</p>

      <h2>Tools and agents: let it act</h2>
      <p>Let the model emit a structured <strong>tool call</strong> — search, calculator, run code, send an
        email — your code executes it and feeds the result back. Wrap that in a loop —
        <strong> think → act → observe → repeat</strong> — and you have an <strong>agent</strong> that takes
        multiple steps toward a goal instead of answering in one shot.</p>

      <div className="callout">A sharp edge to remember: everything retrieved or returned by a tool is
        <em> untrusted data</em>, not instructions. Treating it as instructions is the <strong>prompt-injection</strong>
        risk — and it's most dangerous precisely for agents that <em>act</em> on what they read.</div>
    </>
  ),

  p27: (
    <>
      <Scene icon="🧪">
        <p>Training the mind is only half the job. The other half is the question every parent and every engineer
          must face: <em>is it actually good — and can I trust it?</em> A model that <em>sounds</em> brilliant and
          a model that <em>is</em> brilliant look identical until you measure them properly.</p>
      </Scene>

      <h2>Measuring quality</h2>
      <p><strong>Perplexity</strong> is the most basic gauge: the exponential of the cross-entropy loss — literally
        how "surprised" the model is by real text. Lower is better.</p>
      <MathBlock>{"\\text{perplexity} = e^{\\,\\text{loss}}"}</MathBlock>
      <p>The intuition that makes it click: <strong>perplexity ≈ how many words the model is effectively choosing
        between</strong> at each step. Two extremes by hand:</p>
      <div className="hand">{`perfect model, always sure of the right word:  loss = 0       → e⁰ = 1   (no confusion)
clueless model, 50,000 equally-likely words:    loss = ln(50000) ≈ 10.8  → e¹⁰·⁸ ≈ 50,000`}</div>
      <p>So perplexity 1 means certainty; perplexity 20 means it's about as unsure as someone guessing among 20
        options. Lower is better. Then come task <strong>benchmarks</strong> (MMLU, GSM8K, HumanEval) for
        comparable scores — but beware
        <strong> data contamination</strong>, where the test answers leaked into training and the score becomes
        meaningless. For open-ended answers, an <strong>LLM-as-judge</strong> grades responses against a rubric.</p>

      <h2>Safety: the part you can't skip</h2>
      <p>Even a well-aligned mind faces <strong>hallucination</strong>, <strong>jailbreaks</strong>, and
        <strong> prompt injection</strong>. There's no single fix — you defend in depth: safety training +
        input/output guardrails + retrieval grounding + a human in the loop for high-stakes actions.</p>

      <div className="callout">The throughline of the whole evaluation stage: <strong>never trust one number</strong>.
        Combine several signals, stay suspicious, and keep a human watching the door.</div>
    </>
  ),

  p28: (
    <>
      <Scene icon="⚡">
        <p>Training the mind was a one-time cost. <strong>Serving</strong> it — answering a million people a day,
          fast, without melting your GPUs — is the cost that never ends. The difference between a demo that works
          on your laptop and a product that works for the world is a handful of inference tricks.</p>
      </Scene>

      <h2>The KV-cache: stop redoing your homework</h2>
      <p>Generating text naively, the model re-reads <em>every previous token</em> before producing each new one —
        quadratic, wasteful work. But a past token's Key and Value never change once computed. So we
        <strong> cache</strong> them: each new token computes its own K/V exactly once and reuses the rest. Here's
        the "50×" made concrete for a 100-token answer:</p>
      <div className="hand">{`naive: token 1 reprocesses 1, token 2 reprocesses 2, … token 100 reprocesses 100
       total = 1 + 2 + … + 100 = 5050 token-reads
cached: each token's K/V computed once → 100 token-reads
       5050 / 100 ≈ 50× less work`}</div>

      <h2>Batching &amp; quantization: fill every seat</h2>
      <p><strong>Continuous batching</strong> keeps the GPU busy by slotting new requests in the instant old ones
        finish, instead of waiting for a whole batch. <strong>Quantized inference</strong> (int8/int4) cuts memory
        4–8× with little quality loss. Engines like <strong>vLLM</strong> bundle all of this — paged KV-cache,
        continuous batching, optimised kernels — behind a familiar OpenAI-style API.</p>

      <div className="callout">The KV-cache's memory footprint — growing with sequence length × batch size — is
        usually what really caps your concurrent users and maximum context length, more than raw compute. Know
        your bottleneck.</div>
    </>
  ),

  p29: (
    <>
      <Scene icon="🛰️">
        <p>The story doesn't end at launch — that's where it gets real. A model in a notebook is an experiment; a
          model that thousands of people depend on is a <em>living product</em> in a world that won't stop
          changing. <strong>MLOps</strong> is how you keep the mind alive, healthy, and honest out there.</p>
      </Scene>

      <h2>Track, version, reproduce</h2>
      <ul>
        <li><strong>Experiment tracking</strong> (MLflow, W&amp;B) logs every run's config, metrics, and artifacts
          so "which version was better?" has a real answer.</li>
        <li><strong>Reproducibility</strong> needs fixed random seeds (numpy, torch, <em>and</em> Python), pinned
          library versions, and versioned data — so a result can be rebuilt months later.</li>
        <li><strong>Versioning</strong> ties data + code + weights into one release you can point to and roll back to.</li>
      </ul>

      <h2>Watch for drift</h2>
      <p>A deployed model rots silently when the live world <strong>drifts</strong> away from its training data —
        new slang, new products, new fraud patterns. Monitor the input and output distributions, alert when they
        shift, and retrain on fresh data. The loop never closes; it just comes around again.</p>

      <div className="callout">Treat models like software releases: stage → canary → full rollout, always with a
        one-click path back to the last known-good version. And with that, the mind you raised is out in the
        world — and the journey comes full circle. 🎉</div>
    </>
  ),
};
