import { lazy, Suspense, type ReactNode } from "react";
import DotProduct from "../lessons/DotProduct";
import MatrixTransform from "../lessons/MatrixTransform";
import GradientDescent from "../lessons/GradientDescent";
import NeuralNetPlayground from "../lessons/NeuralNetPlayground";
import LogisticBoundary from "../lessons/LogisticBoundary";
import BackpropWalkthrough from "../lessons/BackpropWalkthrough";
import OptimizerRace from "../lessons/OptimizerRace";
import EmbeddingExplorer from "../lessons/EmbeddingExplorer";
import DecodingPlayground from "../lessons/DecodingPlayground";
import EntropyExplorer from "../lessons/EntropyExplorer";
import OverfittingExplorer from "../lessons/OverfittingExplorer";
import XORNet from "../lessons/XORNet";
import ConvolutionExplorer from "../lessons/ConvolutionExplorer";
import DPOExplorer from "../lessons/DPOExplorer";
import { ARTICLES } from "./articles";
import { M, MathBlock } from "../components/Math";
import { ShowMath } from "../components/Details";
import { ViewSwitcher } from "../components/Controls";
import { Scene, Mission, RealWorld } from "../components/Story";

// 3-D scenes are lazy-loaded so Three.js only ships to the chapters that use it.
const GradientDescent3D = lazy(() => import("../three/GradientDescent3D"));
const Vectors3D = lazy(() => import("../three/Vectors3D"));
const Embeddings3D = lazy(() => import("../three/Embeddings3D"));
const NeuralNet3D = lazy(() => import("../three/NeuralNet3D"));
const MatrixCube3D = lazy(() => import("../three/MatrixCube3D"));
const Loading3D = () => <div className="viz dark" style={{ height: 420, display: "grid", placeItems: "center", color: "#9d94ff" }}>loading 3D…</div>;
const Lazy3D = ({ children }: { children: ReactNode }) => <Suspense fallback={<Loading3D />}>{children}</Suspense>;

// helper: a figure shown as switchable 2-D / 3-D views (persists the choice per id)
const TwoThreeD = ({ id, twoD, threeD, default3D = false }: { id: string; twoD: ReactNode; threeD: ReactNode; default3D?: boolean }) => (
  <ViewSwitcher id={id} views={default3D
    ? [{ key: "3d", label: "3D", node: <Lazy3D>{threeD}</Lazy3D> }, { key: "2d", label: "2D", node: twoD }]
    : [{ key: "2d", label: "2D", node: twoD }, { key: "3d", label: "3D", node: <Lazy3D>{threeD}</Lazy3D> }]} />
);

// Rich content keyed by lesson id: prose articles (from ./articles) + interactive lessons below.
// Part of the one story — "Raising a Mind". Each opens on a scene, then a playground with a mission.
export const CONTENT: Record<string, ReactNode> = {
  ...ARTICLES,
  "p01-dot-product": (
    <>
      <Scene icon="🤝">
        <p>Two people meet. Do they get along? You can't see "agreement" — but you can <em>feel</em> whether
          they're pushing the same way, ignoring each other, or pulling apart. The mind's very first sense does
          exactly this with numbers. Hand it two lists of numbers and it answers with a single number: <strong>how
          much do these two things agree?</strong> That operation is the <strong>dot product</strong>, and it
          quietly powers every neural-network layer and the attention inside every transformer.</p>
      </Scene>

      <h2>Plain words</h2>
      <p>Multiply two equal-length vectors slot-by-slot, then add it all up. Big positive → they point the same
        way (strong agreement). Zero → perpendicular (unrelated). Negative → they oppose each other.</p>

      <h2>By hand, on tiny numbers</h2>
      <div className="hand">{`a = [2, 1, 3]   b = [1, 0, 2]
a · b = (2×1) + (1×0) + (3×2)
      =   2   +   0   +   6   = `}<span className="hl">8</span></div>

      <h2>Play with it — in 2D or 3D</h2>
      <p>Drag the arrowheads. The dot product updates live, next to the angle between the arrows and the
        projection of one onto the other. Flip to <strong>3D</strong> and orbit: real word- and image-vectors
        live in hundreds of dimensions, but the rule never changes — <em>direction becomes a single number</em>.</p>
      <TwoThreeD id="dot" twoD={<DotProduct />} threeD={<Vectors3D />} />
      <Mission>
        <li>Make <code>a · b = 0</code> — find the angle where two ideas become perfectly <em>unrelated</em>.</li>
        <li>Make it <strong>negative</strong> — point the arrows apart and watch agreement turn to opposition.</li>
        <li>Keep the angle fixed but stretch <code>b</code> longer — the dot product grows even though their
          <em> direction</em> never changed. (That's why we'll later divide it out with cosine.)</li>
      </Mission>

      <RealWorld>A song recommender stores your taste as a vector and each track as a vector; the dot product
        scores the match. "You might also like…" is this one number, computed a million times.</RealWorld>

      <div className="callout">Geometrically, <code>a·b = |a|·|b|·cos(θ)</code> — it peaks when the arrows align
        (θ=0°), is zero when perpendicular (θ=90°), and goes negative past 90°.</div>
      <ShowMath>
        <MathBlock>{"\\mathbf{a}\\cdot\\mathbf{b} = \\sum_i a_i b_i = \\lVert\\mathbf{a}\\rVert\\,\\lVert\\mathbf{b}\\rVert\\cos\\theta"}</MathBlock>
      </ShowMath>
    </>
  ),

  "p01-matrix": (
    <>
      <Scene icon="🗺️">
        <p>Stretch a photo on your phone, rotate a map with two fingers, skew a shape in a slideshow — every one
          of those is a grid of numbers reaching into space and <strong>moving every point at once</strong>. That
          grid is a <strong>matrix</strong>, and that move is the entire linear half of what a neural-network layer
          does to its inputs: stretch, shear, rotate, flip them into a more useful arrangement.</p>
      </Scene>

      <h2>Plain words</h2>
      <p>A 2×2 matrix <code>W</code> sends each vector <code>x</code> to a new vector <code>Wx</code>. Watch what
        it does to the whole grid and to the little unit square — that deformation <em>is</em> the transformation.</p>

      <h2>By hand, on tiny numbers</h2>
      <div className="hand">{`W = [1 2]   x = [2]   Wx = [1·2 + 2·1]   [4]
    [0 1]       [1]        [0·2 + 1·1] = `}<span className="hl">[1]</span></div>

      <h2>Play with it — in 2D or 3D</h2>
      <p>Drag the entries of <code>W</code>, or hit a preset, and watch space bend. The arrows show where the
        basis vectors land; <code>det</code> tells you how much area (2D) or volume (3D) was scaled — and it flips
        sign when space is mirrored. Switch to <strong>3D</strong> to orbit a transformed cube.</p>
      <TwoThreeD id="matrix" twoD={<MatrixTransform />} threeD={<MatrixCube3D />} />
      <Mission>
        <li>Find a <strong>rotation</strong>: turn the grid without changing its area (keep <code>det = 1</code>).</li>
        <li>Make <code>det</code> go <strong>negative</strong> — flip space inside-out like a mirror.</li>
        <li>Squash <code>det</code> to <strong>0</strong> — collapse the whole plane onto a line, and watch
          information get destroyed (this is exactly what kills a layer if you're not careful).</li>
      </Mission>

      <div className="callout">Stack a few of these (matrix × matrix) and you have the linear half of every
        neural-network layer. Add a nonlinearity between them — coming in Part 2 — and the mind can bend space
        into <em>any</em> shape it needs.</div>
    </>
  ),

  "p02-gradient-descent": (
    <>
      <Scene icon="🎿">
        <p>Close your eyes on a foggy mountainside and try to reach the valley floor. You can't see the bottom —
          but you can feel which way the ground slopes under your feet, and step downhill. Take a step, feel
          again, step again. That blind, patient descent is <strong>exactly</strong> how every model on Earth
          learns. It's called <strong>gradient descent</strong>, and the size of your step — the
          <strong> learning rate</strong> — is the single most important knob in all of deep learning.</p>
      </Scene>

      <h2>Plain words</h2>
      <p>The rule is <code>w ← w − learning_rate × gradient</code>. The gradient points <em>uphill</em> (toward
        more error), so we step the opposite way. Too small a step and it crawls for hours; too big and it
        overshoots the valley — or flies off the mountain entirely.</p>

      <h2>By hand, on tiny numbers</h2>
      <div className="hand">{`L(w) = (w − 3)²     slope L'(w) = 2(w − 3)
start w=0, lr=0.1:
  grad = 2(0−3) = −6   ->   w = 0 − 0.1·(−6) = `}<span className="hl">0.6</span></div>
      <p>One step moved <code>w</code> from 0 toward 3 (the true bottom). Repeat and it homes in. This is the
        <strong> heartbeat</strong> you'll meet again in regression, in neural nets, and in GPT: <em>predict →
        measure how wrong → step downhill</em>.</p>

      <h2>Roll the ball — in 3D or 2D</h2>
      <p>Loss is really a <em>landscape</em>, and learning is rolling downhill on it. In <strong>3D</strong>, orbit
        the valley and press <strong>Play</strong> (or <strong>Step</strong> one update at a time). In
        <strong> 2D</strong> you watch the same descent on a single slice.</p>
      <TwoThreeD id="gd" default3D twoD={<GradientDescent />} threeD={<GradientDescent3D />} />
      <Mission>
        <li>Set a tiny learning rate (0.01) — feel how <em>painfully slow</em> careful learning can be.</li>
        <li>Crank it to ~1.0 — watch the ball bounce back and forth across the valley instead of settling.</li>
        <li>Push past ~1.05 — watch it <strong>diverge and fly uphill</strong>. You just reproduced the
          single most common training disaster.</li>
      </Mission>

      <div className="callout">That last experiment has a name in the wild: when someone says "my loss went to
        <code> NaN</code>," it almost always means <em>learning rate too high</em> — the ball flew off the
        mountain. Now the mind can chase a goal. Next, we teach it to handle <em>uncertainty</em>.</div>
    </>
  ),

  p03: (
    <>
      <Scene icon="🎲">
        <p>A weather app never says "it <em>will</em> rain." It says "70% chance." A language model is the same:
          it never declares the next word — it spreads its belief across every possible word. To raise a mind
          that thinks in probabilities, we need the language of uncertainty: <strong>distributions</strong>,
          <strong> entropy</strong>, and the <strong>cross-entropy</strong> loss it's actually trained to minimise.</p>
      </Scene>

      <h2>Entropy: how unsure is a guess?</h2>
      <p><strong>Entropy</strong> measures average surprise. It's highest when every outcome is equally likely (a
        fair die — maximum confusion) and zero when one outcome is certain (a loaded die that always lands six):</p>
      <MathBlock>{"H(p) = -\\sum_i p_i \\ln p_i"}</MathBlock>
      <p>Worth doing once by hand so the formula isn't a mystery. Three guesses about a 3-way outcome:</p>
      <div className="hand">{`certain   [1, 0, 0]:        H = −(1·ln1)                       = 0       (no surprise)
lopsided  [0.5, 0.25, 0.25]: H = −(0.5·ln0.5 + 0.25·ln0.25 ×2)  ≈ 1.04
uniform   [⅓, ⅓, ⅓]:        H = −(3 · ⅓·ln⅓) = ln 3            ≈ 1.10    (most surprise)`}</div>
      <p>Drag the bars below — entropy peaks when they're level (≈1.10) and collapses to 0 when one bar dominates.</p>
      <EntropyExplorer />
      <Mission>
        <li>Flatten all the bars equal — push entropy to its <strong>maximum</strong> (total uncertainty).</li>
        <li>Pile everything onto one bar — drive entropy toward <strong>zero</strong> (total certainty).</li>
      </Mission>

      <h2>Cross-entropy: the score the mind chases</h2>
      <p>For next-word prediction the truth is "one-hot" — the correct word has probability 1, the rest 0. So
        cross-entropy collapses into something beautifully simple: the negative log of the probability the model
        gave the <em>right</em> word.</p>
      <MathBlock>{"L = -\\ln p_{\\text{correct}}"}</MathBlock>
      <div className="hand">{`model gives the right word p = 0.9  →  L = −ln(0.9) = 0.11   (confident & right — tiny loss)
model gives the right word p = 0.5  →  L = −ln(0.5) = 0.69
model gives the right word p = 0.1  →  L = −ln(0.1) = 2.30   (it doubted the truth — big loss)`}</div>
      <div className="callout">Confident <em>and</em> right → loss near 0. Confident <em>and</em> wrong → loss
        explodes toward infinity. That punishing shape is exactly what we want, and it's the loss every single
        LLM is trained to minimise. The mind now has a goal, a way to step toward it, and a way to weigh doubt —
        it's ready to eat.</div>
    </>
  ),

  p05: (
    <>
      <Scene icon="🦜">
        <p>A student who memorises last year's exam answers word-for-word looks like a genius — right up until
          the questions change, and they fall apart. This is the mind's very first temptation, and it's so common
          it has a name: <strong>overfitting</strong>. A model that's too complex memorises the training data —
          noise and all — and then fails on anything new. Let's catch it red-handed.</p>
      </Scene>

      <h2>See it happen</h2>
      <p>Below we fit a polynomial to noisy points. Slide the degree up: degree 1 (a straight line) is too simple
        and <em>underfits</em>; degree ~3 captures the true curve; degree 10+ wiggles through every single point —
        its training error drops to almost zero while its <strong>test error explodes</strong>. That gap is the
        memorisation, made visible.</p>
      <OverfittingExplorer />
      <Mission>
        <li>Find the <strong>sweet spot</strong> — the lowest degree that still follows the true curve.</li>
        <li>Push the degree to the max and watch training error fall while test error <em>climbs</em>: the
          telltale signature of a memoriser.</li>
        <li>Drop to degree 1 and feel the opposite failure — too simple to learn the pattern at all.</li>
      </Mission>

      <h2>The takeaway</h2>
      <p>Low training error is <em>not</em> the goal. Low error on <strong>unseen</strong> data is. That's why we
        split data into train / validation / test, and why we regularize. The mind we want is just complex enough
        to capture the real pattern — and no more.</p>
      <div className="callout">This bias–variance balance is the quiet villain behind everything that follows —
        from choosing a model's size to adding dropout and weight decay in deep nets.</div>
    </>
  ),

  p07: (
    <>
      <Scene icon="📧">
        <p>Your inbox makes a thousand silent decisions a day: spam or not spam. Each one is a line drawn through
          a space of features — and on which side of the line an email falls decides its fate. That line-drawer is
          <strong> logistic regression</strong>: the simplest classifier there is, and secretly a single neuron.
          It takes a linear score, squashes it through an S-curve into a probability, then draws a
          <strong> straight</strong> boundary between yes and no.</p>
      </Scene>

      <h2>The model</h2>
      <p>Take a linear score and pass it through the <strong>sigmoid</strong> to get a probability in (0, 1):</p>
      <MathBlock>{"p = \\sigma(\\mathbf{w}\\cdot\\mathbf{x} + b), \\qquad \\sigma(z) = \\frac{1}{1+e^{-z}}"}</MathBlock>
      <p>The sigmoid is the "squash any number into a probability" function. By hand it bends a raw score into (0, 1):</p>
      <div className="hand">{`σ(0)  = 1/(1+e⁰)   = 1/2       = 0.50   ← right on the fence
σ(+2) = 1/(1+e⁻²)  = 1/1.135   = 0.88   ← confident "yes"
σ(−2) = 1/(1+e²)   = 1/8.39    = 0.12   ← confident "no"`}</div>
      <p>So a big positive score → near 1 ("spam"), a big negative → near 0 ("not spam"), and 0 → "I'm unsure, 50/50".
        We train it with binary cross-entropy, and its gradient is gorgeously simple — that same
        "<M>{"(p - y)"}</M> times the input" shape that drives <em>all</em> of learning:</p>
      <ShowMath>
        <MathBlock>{"\\frac{\\partial L}{\\partial \\mathbf{w}} = (p - y)\\,\\mathbf{x}"}</MathBlock>
      </ShowMath>

      <h2>Watch it train — and watch it fail</h2>
      <p>Press <strong>Train</strong>. On the <strong>Gaussian</strong> blobs the boundary snaps into place at
        ~100% accuracy. Now switch to <strong>Circle</strong> or <strong>XOR</strong>: it stalls — because
        <em> no single straight line</em> can ever separate those classes.</p>
      <LogisticBoundary />
      <Mission>
        <li>Train on <strong>Gaussian</strong> and reach ~100% — see a straight cut do its job perfectly.</li>
        <li>Switch to <strong>Circle</strong> and watch it flail. One line can't surround a blob.</li>
        <li>Try <strong>XOR</strong> — the failure that, historically, nearly killed neural networks (until the
          next chapter rescued them).</li>
      </Mission>

      <div className="callout">That failure is the <em>whole motivation</em> for neural networks: stack these
        units with a nonlinearity between them and the boundary can bend into any shape — exactly what you'll do
        in the <strong>Neural Network Playground</strong>.</div>
    </>
  ),

  p10: (
    <>
      <Scene icon="🧠">
        <p>One neuron can only draw a straight line — we just watched it fail on XOR. But your brain isn't one
          neuron; it's billions, wired together. So we do the same thing: take many tiny straight-line decisions,
          stack them in layers with a nonlinearity between, and something remarkable happens. A <strong>brain
          appears</strong> that can carve curves no single neuron ever could.</p>
      </Scene>

      <h2>Meet the network</h2>
      <p>Here it is: neurons stacked in layers, each feeding the next. Orbit it and watch the activation flow
        left→right — that's a <strong>forward pass</strong>, the signal turning input into output.</p>
      <Lazy3D><NeuralNet3D /></Lazy3D>

      <h2>Why a hidden layer changes everything</h2>
      <p>XOR is the classic proof. Its two classes sit on opposite diagonals, so <em>no straight line</em> can
        separate them — a single neuron is stuck forever. Add a hidden layer and the network can <strong>combine
        several lines into a curve</strong>.</p>
      <XORNet />
      <Mission>
        <li>Start with <strong>0 hidden neurons</strong> — confirm the net is just a line, and fails (it's back to
          logistic regression).</li>
        <li>Add neurons one at a time — watch the faint grey lines (each hidden neuron's own boundary) appear and
          bend together into a curve that finally cracks XOR.</li>
        <li>Switch <strong>tanh ↔ ReLU</strong> and feel how differently the boundary forms.</li>
      </Mission>
      <ShowMath>
        <MathBlock>{"\\text{layer: } a = f(Wx + b), \\quad f = \\text{ReLU or tanh}"}</MathBlock>
      </ShowMath>

      <p>Why is that nonlinearity <M>{"f"}</M> non-negotiable? Watch what happens <em>without</em> it — two plain
        linear layers, by hand:</p>
      <div className="hand">{`layer 1:  h = 2·x          layer 2:  y = 3·h
stack them:  y = 3·(2·x) = 6·x   ← that's just ONE layer with weight 6`}</div>
      <p>Two layers collapsed into one. A hundred linear layers would too. Depth buys you <em>nothing</em> until
        you bend the signal between layers with a nonlinearity (ReLU, tanh) — only then can each layer add a new
        fold and the network trace a curve.</p>
      <div className="callout byhand">That fold is the whole point. Next: <em>how</em> does this brain actually
        learn its weights?</div>
    </>
  ),

  p11: (
    <>
      <Scene icon="💡">
        <p>This is the most important idea in the entire journey — the flash of learning itself. A network makes
          a guess, sees how wrong it was, and then asks the only question that matters: <em>which of my thousands
          of knobs should I turn, and which way, to be a little less wrong next time?</em> The algorithm that
          answers it is <strong>backpropagation</strong> — and, stripped of its scary name, it is nothing but the
          <strong> chain rule</strong> run backward, reusing the numbers from the forward pass.</p>
      </Scene>

      <h2>The tiniest learnable thing</h2>
      <p>We use the smallest unit that can learn: one weight, one bias, a sigmoid, and a squared-error loss.</p>
      <MathBlock>{"z = w x + b \\;\\to\\; a = \\sigma(z) \\;\\to\\; L = (a - y)^2"}</MathBlock>

      <h2>Forward, then backward</h2>
      <p><strong>Forward</strong> (blue, left→right) builds the loss. <strong>Backward</strong> (red, right→left)
        sends a gradient back through each node — and at every node we just multiply the <em>local</em> slope by
        the gradient handed to us. That multiply <em>is</em> the chain rule:</p>
      <ShowMath>
        <MathBlock>{"\\frac{\\partial L}{\\partial w} = \\underbrace{2(a-y)}_{\\partial L/\\partial a}\\cdot \\underbrace{a(1-a)}_{\\partial a/\\partial z}\\cdot \\underbrace{x}_{\\partial z/\\partial w}"}</MathBlock>
      </ShowMath>
      <p>Let's run the whole forward-then-backward pass on real numbers, so "the chain rule" becomes three
        multiplications you can check:</p>
      <div className="hand">{`forward:  x=1, w=0.5, b=0, y=0
  z = w·x + b = 0.5      a = σ(0.5) = 0.62      L = (0.62 − 0)² = 0.39
backward (multiply the three local slopes):
  ∂L/∂a = 2(a−y)   = 2·0.62        = 1.24
  ∂a/∂z = a(1−a)   = 0.62·0.38     = 0.24
  ∂z/∂w = x        = 1
  ∂L/∂w = 1.24 × 0.24 × 1 = 0.30   ← nudge w down by lr·0.30 to cut the loss`}</div>
      <BackpropWalkthrough />
      <Mission>
        <li>Step the graph <strong>forward</strong> and read off the loss — then step <strong>backward</strong>
          and watch a gradient appear at every node.</li>
        <li>Hit <strong>"Take a gradient step"</strong> over and over and watch <M>{"L"}</M> shrink as
          <M>{" w"}</M> and <M>{"b"}</M> update. You're watching learning happen, one knob-turn at a time.</li>
      </Mission>

      <div className="callout byhand">Scale this exact procedure to billions of parameters and you have
        <code> loss.backward()</code> in PyTorch. There is no extra magic — only the chain rule, repeated.</div>
    </>
  ),

  p12: (
    <>
      <Scene icon="🏎️">
        <p>Backprop tells the mind <em>which way</em> is downhill. But how big a step should it take, and should
          it build momentum like a ball rolling, or tiptoe on steep ground? That decision belongs to the
          <strong> optimizer</strong>. Plain gradient descent zig-zags pathetically across stretched valleys —
          and the fix for that is why <strong>Adam</strong> trains very nearly every modern model.</p>
      </Scene>

      <h2>The four contenders</h2>
      <ul>
        <li><strong>SGD</strong> — step straight downhill: <M>{"w \\leftarrow w - \\eta\\,g"}</M>. Honest, but
          easily flummoxed.</li>
        <li><strong>Momentum</strong> — accumulate velocity, so consistent directions build speed (the rolling ball).</li>
        <li><strong>RMSProp</strong> — scale each parameter's step by its recent gradient size (tiptoe where it's steep).</li>
        <li><strong>Adam</strong> — momentum <em>and</em> per-parameter scaling, with bias correction. The
          workhorse.</li>
      </ul>

      <h2>Race them</h2>
      <p>On the <strong>stretched ravine</strong> the loss is far steeper across the valley than along it. Watch
        plain SGD (grey) bounce wall-to-wall while Adam (violet) and RMSProp (green) cut straight to the bottom.</p>
      <OptimizerRace />
      <Mission>
        <li>Race all four on the ravine — see who reaches the minimum first, and who wastes its energy bouncing.</li>
        <li>Nudge the learning rate up until even Adam starts to overshoot — every optimizer has a breaking point.</li>
      </Mission>

      <div className="callout">Pair Adam with a learning-rate schedule (a warmup, then cosine decay) and you have
        the recipe used to train essentially every transformer in existence.</div>
    </>
  ),

  playground: (
    <>
      <Scene icon="🧪">
        <p>Enough watching — step into the lab. Below is a real neural network, built from scratch and training
          <strong> live in your browser</strong>. No server, no library: just the forward pass and the
          backpropagation you just met, running on toy data. This is the playground where everything you've
          learned becomes something you can touch.</p>
      </Scene>

      <h2>What you're seeing</h2>
      <p>The dots are data in two classes (<span style={{ color: "#2563eb", fontWeight: 700 }}>blue</span> /
        <span style={{ color: "#e0533d", fontWeight: 700 }}> red</span>). The shaded background is the network's
        current <strong>decision boundary</strong> — where it switches its guess. As it trains, watch the
        boundary bend to fit the data in real time.</p>
      <NeuralNetPlayground />
      <Mission>
        <li><strong>Circle</strong> and <strong>Gaussian</strong> train fast — a shallow net handles them easily.</li>
        <li>Try <strong>Spiral</strong> with one tiny layer and watch it <em>fail</em>; then add neurons and layers
          until it succeeds. That's the raw power of depth and nonlinearity, in your hands.</li>
        <li>Flip <strong>tanh ↔ ReLU</strong> and feel the boundaries form differently.</li>
        <li>Crank the <strong>learning rate</strong> too high and watch training shake itself apart — the same
          divergence from the gradient-descent chapter, now in a full network.</li>
      </Mission>

      <div className="callout byhand">Everything here — the layers, the activations, the gradient steps — is
        exactly what you stepped through in <strong>Backpropagation</strong> and <strong>Optimizers</strong>.
        This is those ideas, made tangible.</div>
    </>
  ),

  p15: (
    <>
      <Scene icon="👁️">
        <p>Hold up a photo of a cat. You don't inspect it pixel by pixel — your eyes hunt for <em>edges</em>, then
          textures, then ears and whiskers, then "cat." We give the mind the same gift of sight with a
          <strong> convolutional network</strong>: it slides small <strong>filters</strong> across an image,
          spotting the same little pattern everywhere it appears — with a tiny fraction of the parameters it would
          take to wire up every pixel.</p>
      </Scene>

      <h2>Convolution = sliding dot products</h2>
      <p>A filter (kernel) is a small grid of weights. At every position it computes the dot product with the
        patch of image it covers — there's our Part-0 sense again — producing a <strong>feature map</strong> that
        lights up wherever the pattern occurs. Here's an edge-detector kernel <code>[−1, 0, +1]</code> by hand, on
        a patch that straddles an edge versus a flat patch:</p>
      <div className="hand">{`edge patch  [10, 10, 80]:  (−1·10) + (0·10) + (1·80) = −10 + 0 + 80 = 70   ← BRIGHT: found an edge
flat patch  [10, 10, 10]:  (−1·10) + (0·10) + (1·10) = −10 + 0 + 10 =  0   ← dark: nothing here`}</div>
      <p>The kernel fires only where the pixels jump — that's edge detection from one tiny dot product, slid
        everywhere.</p>
      <ConvolutionExplorer />
      <Mission>
        <li>Pick an <strong>edge</strong> kernel and slide it — watch the feature map glow along boundaries and
          go dark on flat regions.</li>
        <li>Swap kernels and see how each one hunts a different pattern (vertical edges, blur, sharpen).</li>
        <li>Notice the same filter runs <em>everywhere</em> — learn a detector once, find the pattern anywhere.</li>
      </Mission>

      <h2>Why it wins on images</h2>
      <p>Two ideas do all the work: <strong>locality</strong> (features like edges are local) and <strong>weight
        sharing</strong> (the same filter sweeps the whole image, so a pattern learned in one spot is found
        anywhere — and the parameter count stays tiny). Stack convolution → ReLU → pooling and the network builds
        edges → textures → object parts → objects.</p>
      <div className="callout">We hand-set the kernels here so you can see the mechanism. A real CNN
        <em> learns</em> its filters by backprop — discovering whatever detectors the task secretly needs.</div>
    </>
  ),

  p18: (
    <>
      <Scene icon="🧭">
        <p>How do you explain to a machine that "king" and "queen" are cousins, while "king" and "banana" are
          strangers? You can't define it — but you can <em>place</em> it. Give every word a home in space, learned
          so that words used in similar contexts end up as neighbours, and something magical falls out:
          <strong> meaning becomes geometry</strong>. Similarity turns into an angle, and relationships turn into
          directions you can literally point along.</p>
      </Scene>

      <h2>Similarity = cosine of the angle</h2>
      <p>Relatedness is measured by <strong>cosine similarity</strong> — the dot product divided by both lengths,
        so only <em>direction</em> matters (not how long the vectors happen to be):</p>
      <MathBlock>{"\\cos(\\mathbf{a},\\mathbf{b}) = \\frac{\\mathbf{a}\\cdot\\mathbf{b}}{\\lVert\\mathbf{a}\\rVert\\,\\lVert\\mathbf{b}\\rVert}"}</MathBlock>
      <p>Cosine ranges from +1 (same direction → synonyms) through 0 (perpendicular → unrelated) to −1 (opposite).
        By hand on two tiny "word" vectors:</p>
      <div className="hand">{`a = [1, 1]   b = [1, 0]
a·b = 1·1 + 1·0 = 1     |a| = √(1²+1²) = 1.41     |b| = 1
cos = 1 / (1.41 · 1) = 0.71   → 45° apart: fairly related`}</div>
      <p>Click any two words below. Switch to <strong>3D</strong> and orbit the word-space — animals cluster in
        one region, royalty in another, and the analogy arrows reveal the hidden directions.</p>
      <TwoThreeD id="emb" twoD={<EmbeddingExplorer />} threeD={<Embeddings3D />} />
      <Mission>
        <li>Click a same-theme pair (cat · dog) — high cosine. Then a cross-theme pair (cat · king) — low. Feel
          meaning become a number.</li>
        <li>Hunt for the <strong>"gender" direction</strong>: the arrow from <em>man</em> to <em>woman</em> matches
          the arrow from <em>king</em> to <em>queen</em>.</li>
      </Mission>

      <h2>Relationships are directions</h2>
      <p>Because that gender arrow is the same everywhere, the famous analogy falls straight out of vector
        arithmetic:</p>
      <MathBlock>{"\\text{king} - \\text{man} + \\text{woman} \\approx \\text{queen}"}</MathBlock>
      <div className="callout">A transformer goes one crucial step further: with attention it makes each word's
        vector <em>context-dependent</em>, so "bank" lands in a different spot in "river bank" than in "bank
        account". Static meaning → living meaning.</div>
    </>
  ),

  p24: (
    <>
      <Scene icon="⚖️">
        <p>After finishing school (SFT), the mind follows instructions — but real helpfulness is subtler than
          right-vs-wrong. Faced with two perfectly valid answers, which one would a <em>human</em> actually
          prefer: the warmer one? the more concise one? the safer one? Teaching that taste is the job of
          <strong> RLHF</strong> and its elegant shortcut <strong>DPO</strong>, learned from nothing but pairwise
          comparisons: "answer A is better than B."</p>
      </Scene>

      <h2>Preference as a loss</h2>
      <p>The Bradley–Terry model turns two answers' scores into a probability that one is preferred, via the
        sigmoid. <strong>DPO</strong> optimises the model directly against this single loss — no separate reward
        model, no fragile reinforcement-learning loop:</p>
      <MathBlock>{"L = -\\ln \\sigma\\!\\big(\\beta\\,[(\\Delta_w) - (\\Delta_l)]\\big)"}</MathBlock>
      <p>where <M>{"\\Delta"}</M> is how much more likely the model makes an answer than the frozen reference
        model. Two cases by hand (with <M>{"\\beta = 1"}</M>) show why the loss falls only when the
        <em> winning</em> answer is favoured:</p>
      <div className="hand">{`model prefers the winner   Δw=1.0, Δl=−0.5:  margin = 1.5   σ(1.5) = 0.82  →  L = −ln 0.82 = 0.20  (low, good)
model prefers the loser    Δw=−0.5, Δl=1.0:  margin = −1.5  σ(−1.5) = 0.18 →  L = −ln 0.18 = 1.70  (high, punished)`}</div>
      <p>Tune the sliders and watch the loss respond.</p>
      <DPOExplorer />
      <Mission>
        <li>Push the <strong>winner's</strong> Δ above the <strong>loser's</strong> — watch the margin go positive
          and the loss fall. That's the model learning to prefer the better answer.</li>
        <li>Crank <M>{"\\beta"}</M> up and down — feel how it tightens or loosens the leash to the reference model.</li>
      </Mission>

      <div className="callout"><M>{"\\beta"}</M> is the leash that stops the model drifting too far from the
        reference — the same idea as the KL penalty in PPO, baked into one clean objective. Next, we learn the
        dials that shape its actual <em>voice</em>.</div>
    </>
  ),

  p26: (
    <>
      <Scene icon="🎛️">
        <p>The same musician can play a lullaby or a thrash solo. A language model is the same instrument: under
          the hood it outputs a probability for <em>every</em> possible next token, and <strong>decoding</strong>
          is how we turn that cloud of probabilities into actual words. Choose well and it writes fluently;
          choose badly and it either loops robotically or descends into nonsense. These are the dials that set its
          mood — from cautious and factual to wild and creative.</p>
      </Scene>

      <h2>Temperature: the creativity dial</h2>
      <p>Before sampling, we divide the logits by a temperature <M>{"T"}</M> and re-softmax:</p>
      <MathBlock>{"p_i = \\mathrm{softmax}(z_i / T)"}</MathBlock>
      <p>See it bite on the same three logits <code>z = [2, 1, 0]</code> — dividing by <M>{"T"}</M> before softmax
        either exaggerates or evens out the gaps:</p>
      <div className="hand">{`T = 0.5 (cold):  softmax([4, 2, 0])   = [0.87, 0.12, 0.02]   ← sharp, almost always the top token
T = 1.0 (normal): softmax([2, 1, 0])   = [0.67, 0.24, 0.09]
T = 2.0 (hot):   softmax([1, 0.5, 0])  = [0.51, 0.31, 0.19]   ← flat, the long shots get a real chance`}</div>
      <p>Low <M>{"T"}</M> sharpens toward the single most likely token (greedy, safe, repetitive); high
        <M>{" T"}</M> flattens the distribution (diverse, surprising, risky).</p>

      <h2>Truncation: top-k and top-p</h2>
      <p><strong>Top-k</strong> keeps only the <M>{"k"}</M> highest-probability tokens; <strong>top-p</strong>
        (nucleus) keeps the smallest set whose probabilities sum to <M>{"p"}</M>. Both ban the unlikely "junk"
        tokens while preserving variety. Play with all three, then hit sample:</p>
      <DecodingPlayground />
      <Mission>
        <li>Set temperature near <strong>0</strong> and sample repeatedly — watch it give the same safe answer
          every time.</li>
        <li>Crank temperature high — watch the junk tokens creep in, then rein them back with <strong>top-p</strong>.</li>
        <li>Find a setting that's creative <em>without</em> going incoherent (hint: top-p ≈ 0.9).</li>
      </Mission>

      <div className="callout">Rule of thumb: low temperature for facts and code, higher for brainstorming;
        top-p ≈ 0.9 is a strong, adaptive default that most production systems lean on.</div>
    </>
  ),
};
