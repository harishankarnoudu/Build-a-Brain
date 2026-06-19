# TinyGPT From Scratch Learning Roadmap

## Objective

Build a GPT-style language model completely from scratch inside Jupyter Notebooks while understanding every component, mathematical operation, tensor transformation, and training step.

The goal is not to create a powerful model initially. The goal is to understand how modern LLMs work internally before moving to fine-tuning production models such as Llama, Mistral, or Qwen.

---

# Project Structure

```text
TinyGPT-Learning/

├── notebooks/
│   ├── 00_Introduction.ipynb
│   ├── 01_Tokenizer.ipynb
│   ├── 02_Embeddings.ipynb
│   ├── 03_Neural_Network_Basics.ipynb
│   ├── 04_Self_Attention.ipynb
│   ├── 05_MultiHead_Attention.ipynb
│   ├── 06_Transformer_Block.ipynb
│   ├── 07_Positional_Encoding.ipynb
│   ├── 08_Build_TinyGPT.ipynb
│   ├── 09_Train_TinyGPT.ipynb
│   ├── 10_Text_Generation.ipynb
│   ├── 11_Model_Interpretability.ipynb
│   ├── 12_SecurityGPT.ipynb
│   └── 13_Modern_LLM_FineTuning.ipynb
│
├── data/
│
├── models/
│
├── outputs/
│
└── README.md
```

---

# Development Rules

For every notebook:

1. Explain the theory first.
2. Explain the mathematics.
3. Explain the tensor dimensions.
4. Print every important variable.
5. Visualize intermediate outputs whenever possible.
6. Show expected output.
7. Show common mistakes.
8. Include exercises.
9. Include debugging explanations.
10. Never hide implementation details.

Every notebook should be understandable by someone learning deep learning from first principles.

---

# Notebook 00 - Introduction

## Goals

Understand:

* What is AI
* What is Machine Learning
* What is Deep Learning
* What is a Neural Network
* What is a Transformer
* What is an LLM
* Why GPT works

## Visualizations

Create diagrams showing:

```text
Text
↓
Tokens
↓
Embeddings
↓
Attention
↓
Transformer Layers
↓
Prediction
```

## Deliverables

* Conceptual overview
* Architecture diagrams
* Tensor flow explanation

---

# Notebook 01 - Build a Tokenizer

## Goals

Understand:

* Vocabulary
* Tokenization
* Encoding
* Decoding

## Tasks

Implement:

### Word-Level Tokenizer

Example:

```text
Hello AI World
```

Vocabulary:

```python
{
    "Hello":0,
    "AI":1,
    "World":2
}
```

Encoded:

```python
[0,1,2]
```

### Character-Level Tokenizer

Example:

```python
"hello"
```

Encoded:

```python
[7,4,11,11,14]
```

## Visualizations

Display:

* Vocabulary size
* Token frequency
* Encoding examples

## Deliverables

Fully working tokenizer from scratch.

---

# Notebook 02 - Embeddings

## Goals

Understand:

* Why tokens become vectors
* Embedding matrices
* Semantic similarity

## Tasks

Implement:

```python
nn.Embedding()
```

Then manually build embedding lookup logic.

## Visualizations

Display:

* Embedding matrix
* Individual token vectors
* Similarity calculations

## Deliverables

Token -> Vector conversion system.

---

# Notebook 03 - Neural Network Fundamentals

## Goals

Understand:

* Linear layers
* Activations
* Loss functions
* Backpropagation

## Tasks

Build:

```text
Input
↓
Linear
↓
ReLU
↓
Linear
↓
Softmax
```

Predict next word.

## Deliverables

Mini language model without attention.

---

# Notebook 04 - Self Attention

## Goals

Understand:

* Query
* Key
* Value
* Attention Scores

## Tasks

Implement attention manually.

Compute:

Q = XWq

K = XWk

V = XWv

Calculate attention scores step-by-step.

Print every matrix.

## Visualizations

Heatmaps for:

* Attention scores
* Softmax output
* Final attention matrix

## Deliverables

Self-attention implementation from scratch.

---

# Notebook 05 - Multi-Head Attention

## Goals

Understand:

* Why multiple heads exist
* Parallel attention

## Tasks

Create:

* 1 head
* 2 heads
* 4 heads

Compare outputs.

## Visualizations

Separate heatmap per attention head.

## Deliverables

Complete multi-head attention module.

---

# Notebook 06 - Transformer Block

## Goals

Understand:

* Residual connections
* Layer normalization
* Feed-forward networks

## Tasks

Build:

```text
Input
↓
Attention
↓
Residual
↓
LayerNorm
↓
FeedForward
↓
Residual
↓
LayerNorm
```

## Deliverables

Complete transformer block.

---

# Notebook 07 - Positional Encoding

## Goals

Understand:

* Why transformers need positions
* How order is preserved

## Tasks

Implement:

* Learned positional embeddings
* Sinusoidal positional embeddings

## Visualizations

Plot positional encodings.

## Deliverables

Position-aware token representations.

---

# Notebook 08 - Build TinyGPT

## Goals

Assemble all previous components.

## Architecture

```text
Token Embedding
+
Position Embedding
↓
Transformer Block x N
↓
Language Modeling Head
↓
Next Token Prediction
```

## Suggested Configuration

```python
vocab_size = 5000

embedding_dim = 128

num_heads = 4

num_layers = 4

context_length = 128
```

Target size:

```text
2M - 10M parameters
```

## Deliverables

Fully functional GPT architecture.

---

# Notebook 09 - Train TinyGPT

## Goals

Train the model.

## Dataset

Start with:

* Small Wikipedia subset
* Security articles
* Documentation

Maximum:

```text
10 MB - 100 MB
```

for initial experiments.

## Tasks

Implement:

* Dataset loader
* Batching
* Training loop
* Checkpoint saving

## Metrics

Track:

* Loss
* Learning rate
* Training speed

## Deliverables

Trained TinyGPT model.

---

# Notebook 10 - Text Generation

## Goals

Generate text.

## Tasks

Implement:

### Greedy Decoding

### Temperature Sampling

### Top-K Sampling

### Top-P Sampling

## Visualizations

Show token probabilities.

## Deliverables

Interactive text generation notebook.

---

# Notebook 11 - Model Interpretability

## Goals

Understand what the model learned.

## Tasks

Visualize:

* Attention maps
* Embedding clusters
* Token relationships

## Questions

Answer:

* Why was this token predicted?
* Which words are similar?
* Which tokens influence others?

## Deliverables

Model inspection toolkit.

---

# Notebook 12 - SecurityGPT

## Goals

Create a security-focused model.

## Training Data

Collect:

* MITRE ATT&CK
* CVE descriptions
* Security blogs
* Sigma rules
* YARA rules
* Malware reports

## Tasks

Train TinyGPT on security corpus.

Evaluate:

* Security knowledge
* Threat understanding
* Detection reasoning

## Deliverables

TinySecGPT model.

---

# Notebook 13 - Modern LLM Fine-Tuning

## Goals

Transition from TinyGPT to production-grade models.

## Models

Experiment with:

* Llama
* Mistral
* Qwen

## Topics

Learn:

* PEFT
* LoRA
* QLoRA
* Quantization
* RLHF
* RAG
* vLLM

## Deliverables

Successfully fine-tune an existing open-source model.

---

# Final Learning Outcomes

By completing this roadmap, the learner should be able to explain:

1. How tokenization works.
2. How embeddings are learned.
3. How attention operates mathematically.
4. How transformer blocks function.
5. How backpropagation updates model weights.
6. How text generation works.
7. Why hallucinations occur.
8. How fine-tuning modifies model behavior.
9. How modern LLMs are trained.
10. How to build AI security models on top of transformer architectures.

---

# Important Constraint

The implementation should prioritize understanding over optimization.

Whenever there is a choice between:

* Simplicity vs Performance

Choose Simplicity.

Every operation should be visible, inspectable, and explainable inside the notebook.
