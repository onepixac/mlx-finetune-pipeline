# MLX Fine-Tuning Pipeline

**Complete CPT → SFT → GRPO pipeline for fine-tuning language models on Apple Silicon.**

Created by [onepixac](https://github.com/onepixac) with [Claude Opus 4.6](https://anthropic.com).
Based on the paper: [arXiv:2509.13081](https://arxiv.org/abs/2509.13081) — "Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO"

---

## What is this?

This is a set of scripts that lets you **teach a language model new things** on a Mac. No NVIDIA GPU needed. No cloud. Everything runs locally on Apple Silicon.

The pipeline has 3 stages:

### Stage 1: CPT (Continued Pre-Training)
**What it does:** Makes the model read raw text from your domain so it gets familiar with the vocabulary and patterns.

**Example:** You have a medical textbook. CPT makes the model read it cover to cover, so it learns medical terms before you ask it questions.

**Input:** Text files in JSON format: `{"text": "Your raw domain text..."}`

### Stage 2: SFT (Supervised Fine-Tuning)
**What it does:** Teaches the model how to respond to questions. You show it thousands of examples: "when the user says X, respond with Y."

**Example:** `"What is aspirin?" → "Aspirin is a nonsteroidal anti-inflammatory drug used to treat pain..."` — the model learns this pattern across thousands of question-answer pairs.

**Input:** Chat pairs: `{"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}`

### Stage 3: GRPO (Group Relative Policy Optimization)
**What it does:** Improves the quality of responses after SFT. It generates multiple candidate answers, scores them, and reinforces the best ones. Think of it as quality control.

**How it works:**
1. The model generates 4 candidate responses to the same question
2. A separate evaluator (sentence-transformer) scores each one
3. The model learns to produce more responses like the best ones

**Input:** Same dataset as SFT. The GRPO script handles everything.

---

## Why MLX? What are MLX models?

[MLX](https://github.com/ml-explore/mlx-examples) is Apple's machine learning framework for Apple Silicon. It's the equivalent of PyTorch/CUDA but for Mac GPUs.

**The key advantage:** Apple Silicon has **unified memory**. The CPU and GPU share the same RAM. A Mac with 64GB RAM can fine-tune models that would need expensive NVIDIA GPUs with 80GB VRAM, because there's no memory copy between CPU and GPU.

### Regular models vs MLX models

Most models on HuggingFace are in PyTorch format — they need NVIDIA GPUs with CUDA. **MLX models** are the same weights converted to Apple's format so they run on Mac GPUs.

Where to find MLX models: **[mlx-community](https://huggingface.co/mlx-community)** on HuggingFace. They convert popular models to MLX format. Look for models ending in `-MLX-4bit` or `-4bit`.

### Which model to choose?

| Model | Parameters | 4-bit Size | Min RAM | Best for |
|---|---|---|---|---|
| [Gemma 4 E2B](https://huggingface.co/mlx-community/gemma-4-2b-a2b-it-4bit) | 2B | ~1.5 GB | 16 GB | Quick experiments |
| [Qwen3.5-4B](https://huggingface.co/mlx-community/Qwen3.5-4B-MLX-4bit) | 4B | ~2.5 GB | 16 GB | Small tasks |
| [Gemma 4 12B-A4B](https://huggingface.co/mlx-community/gemma-4-12b-a4b-it-4bit) | 4B active | ~7 GB | 24 GB | Good balance |
| [Qwen3.5-9B](https://huggingface.co/mlx-community/Qwen3.5-9B-MLX-4bit) | 9B | ~5.6 GB | 24 GB | Strong all-rounder |
| [Gemma 4 26B-A4B](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) | 4B active (26B total) | ~15.6 GB | 32 GB | Best quality per token |
| [Qwen3.5-32B](https://huggingface.co/mlx-community/Qwen3.5-32B-MLX-4bit) | 32B | ~18 GB | 48 GB | Maximum quality |
| [Llama 3.3-70B](https://huggingface.co/mlx-community/Llama-3.3-70B-Instruct-4bit) | 70B | ~40 GB | 64 GB | Frontier |

**Rule of thumb:** Model size (4-bit) + 8 GB for training overhead = minimum RAM you need.

**"Active" parameters:** Some models like Gemma 4 use Mixture of Experts (MoE). They have 26B total parameters but only 4B are active for each token. This means high quality with lower computation cost.

---

## Hardware Tested

| Mac | RAM | Largest Model | Training Speed | Notes |
|---|---|---|---|---|
| Mac Mini M4 Pro | 64 GB | Gemma 4 26B-A4B | ~0.5 it/sec | Recommended setup |
| MacBook Pro M3 Max | 48 GB | Qwen3.5-32B | ~0.4 it/sec | Works well |
| Mac Studio M2 Ultra | 192 GB | 70B+ models | ~0.3 it/sec | Maximum capacity |
| MacBook Air M2 | 24 GB | Qwen3.5-9B | ~0.2 it/sec | Possible but slow |
| Any M1+ Mac | 16 GB | Gemma 4 E2B | ~0.3 it/sec | Minimum viable |

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/onepixac/mlx-finetune-pipeline.git
cd mlx-finetune-pipeline

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install mlx-lm sentence-transformers

# 4. For Gemma 4 support (required as of April 2026):
pip install 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@main'

# 5. Download a model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/gemma-4-26b-a4b-it-4bit', local_dir='models/gemma4-26b-4bit')
"
```

---

## Usage

### Option 1: Run the full pipeline

```bash
./run_pipeline.sh
```

### Option 2: Run each stage manually

**Stage 1 — CPT:**
```bash
python -m mlx_lm lora -c config_cpt.yaml
```

**Stage 2 — SFT:**
```bash
python -m mlx_lm lora -c config_sft.yaml
```

**Stage 3 — GRPO:**
```bash
# First: fuse SFT adapter into base model (required — see Known Issues)
python -m mlx_lm fuse \
  --model models/your-base-model \
  --adapter-path adapters/stage2-sft \
  --save-path models/fused-sft

# Then: run GRPO
python grpo_train.py
```

---

## Preparing Your Data

### CPT data (raw text)

Create `data/cpt/train.jsonl` with one JSON object per line:

```json
{"text": "Your raw domain text. This can be entire articles, book chapters, documentation, or any text the model should learn to understand."}
{"text": "Another chunk of domain text. Break long documents into chunks of 500-2000 tokens."}
```

### SFT data (question-answer pairs)

Create `data/sft/train.jsonl`:

```json
{"messages": [{"role": "system", "content": "You are an expert medical assistant."}, {"role": "user", "content": "What are the side effects of aspirin?"}, {"role": "assistant", "content": "Common side effects include stomach irritation, heartburn, and nausea. Serious but rare side effects include..."}]}
```

You also need `data/sft/valid.jsonl` (5% of data) and `data/sft/test.jsonl` (5% of data) in the same format.

### The most important lesson we learned

**Your dataset format matters more than its size.**

We trained a model with 12,000 pairs in the wrong format — it invented words and couldn't hold a conversation. We then trained with 5,000 pairs in the right format — it responded correctly.

The wrong format:
```json
{"user": "Read me chapter 1", "assistant": "The text of chapter 1..."}
```
The model learned: when someone says "read me X", recite X. It never learned to converse.

The right format:
```json
{"user": "What happened when the princess looked at the sky?", "assistant": "The text describing what happened..."}
```
Same source material, same response — but the user prompt is a natural question. The model learns: when someone asks about X, respond with the relevant information.

**Rules for good SFT data:**
1. User prompts must be natural questions people would actually ask
2. Never use "read me", "recite", "quote from" as user prompts
3. Balance your dataset: ~40% conversation, ~40% knowledge/translation, ~20% corrections
4. Every assistant response must be authentic — never invent content
5. Deduplicate by assistant response content
6. Manually verify a random sample before training

---

## Configuration Reference

### CPT Config

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | `5e-6` | Low — CPT only adapts, doesn't restructure |
| `max_seq_length` | `1024` | Long chunks for reading comprehension |
| `mask_prompt` | `false` | CPT trains on everything (no system/user distinction) |
| `iters` | `1000-3000` | More isn't better — monitor val loss, stop when it rises |
| `save_every` | `500` | Checkpoint frequently — CPT overfits fast |

### SFT Config

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | `1e-5` | Standard for SFT |
| `max_seq_length` | `512` | Shorter — responses are usually < 256 tokens |
| `mask_prompt` | `true` | Only train on assistant responses, not system/user |
| `iters` | `dataset_size × epochs` | 10 epochs is a good default |
| `save_every` | `5000` | Protect against OOM crashes |

### GRPO Config

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | `1e-6` | Very low — small adjustments only |
| `n_candidates` | `4` | 4 candidates per prompt |
| `n_iters` | `30` | 30 iterations with early stopping |
| `enable_thinking` | `False` | Critical for Qwen3.5 — reward drops from 0.92 to 0.27 with thinking mode |

---

## Known Issues and Solutions

### 1. GRPO crashes with "Can't convert LoRALinear to LoRA"

**Why:** GRPO tries to add LoRA layers on top of an SFT model that already has LoRA layers. LoRA-on-LoRA is not supported.

**Fix:** Fuse the SFT adapter into the base model first:
```bash
python -m mlx_lm fuse \
  --model models/your-base-model \
  --adapter-path adapters/stage2-sft \
  --save-path models/fused-sft
```
Then point GRPO to `models/fused-sft` instead of the base model + adapter.

### 2. Gemma 4 won't load — "510 parameters not in model"

**Why:** The released version of mlx-lm doesn't fully support Gemma 4's MoE architecture.

**Fix:** Install from the main branch:
```bash
pip install 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@main'
```

### 3. Training crashes with OOM (Out of Memory)

**Why:** The model + training overhead exceeds your Mac's RAM. Often caused by running other GPU-heavy tasks in parallel.

**Fix:**
- Set `save_every: 5000` to checkpoint regularly (don't lose hours of work)
- Don't run embedding models or scraping during training
- Reduce `max_seq_length` (512 → 256)
- Use `grad_checkpoint: true`
- Close other GPU-heavy apps

### 4. CPT val loss rises after initial drop

**Why:** The model is overfitting on the small text corpus. Normal behavior.

**Fix:** Don't use the final checkpoint. Use the one with the lowest val loss (usually iter 500-1500). Monitor with:
```bash
grep "Val loss" pipeline_log.txt
```

### 5. Resuming after crash

**Why:** Training crashed and you want to continue from the last checkpoint.

**Fix:**
```bash
python -m mlx_lm lora -c config_sft.yaml \
  --resume-adapter-file adapters/stage2-sft/CHECKPOINT.safetensors \
  --iters REMAINING_ITERS
```
Note: Optimizer state is lost on resume. Val loss will temporarily spike before recovering.

---

## Understanding Training Metrics

### Epochs
An epoch means the model has seen your entire dataset once. If you have 10,000 training pairs and set 10 epochs, the model will see each pair 10 times — that's 100,000 iterations total.

More epochs = the model memorizes better, but too many = **overfitting** (the model memorizes the training data perfectly but can't generalize to new questions).

**Recommended:** 6-10 epochs for SFT. Monitor val loss to decide when to stop.

### Train Loss
This number tells you how well the model is learning the training data. **Lower is better.**

- Starts high (3-7) and drops as the model learns
- Should decrease steadily during training
- If it stops decreasing, the model has learned what it can from the data

### Val Loss (Validation Loss)
This is the most important number. It tells you how well the model performs on data it has **never seen** during training. **Lower is better.**

- Starts high, drops as the model learns to generalize
- At some point it starts rising again — this means **overfitting** (the model is memorizing training data instead of learning general patterns)
- **The checkpoint with the lowest val loss is the best model**

### How checkpoint selection works

mlx-lm automatically saves `adapters.safetensors` with the **best val loss** seen so far. It also saves numbered checkpoints (`0005000_adapters.safetensors`, `0010000_adapters.safetensors`, etc.) at regular intervals.

When val loss is evaluated (every `steps_per_eval` iterations):
- If the new val loss is lower than any previous → `adapters.safetensors` is updated (best model)
- The numbered checkpoint is always saved regardless

**You should always use `adapters.safetensors`** (the best) for the next stage, not the last numbered checkpoint.

### Example training progression

```
Iter  1000: Train loss 3.20, Val loss 2.79  ← model is learning
Iter  2000: Train loss 2.55, Val loss 2.66  ← still improving (val loss dropped)
Iter  3000: Train loss 2.10, Val loss 2.90  ← val loss RISING → overfitting started
                                              Best checkpoint: iter 2000 (val 2.66)
```

In this example, the best model is at iter 2000 even though training continued to iter 3000. The model at iter 3000 has a lower train loss (2.10) but higher val loss (2.90) — it memorized the training data but got worse at generalizing.

---

## How it works (for the curious)

### LoRA (Low-Rank Adaptation)
Instead of updating all 26 billion parameters (which would need terabytes of memory), LoRA adds small trainable matrices to specific layers. Typically only 0.1-4% of parameters are trainable. This is why fine-tuning works on a Mac — you're only training ~100-200 million parameters.

### 4-bit Quantization
Models are stored with 4 bits per parameter instead of 16 or 32. This reduces a 26B model from ~52 GB to ~15 GB, making it fit in Mac RAM. Quality loss is minimal for fine-tuning.

### Unified Memory
On NVIDIA systems, the GPU has its own memory (VRAM). Data must be copied between CPU RAM and GPU VRAM — this is slow and limits model size to VRAM capacity. On Apple Silicon, CPU and GPU share the same memory pool. A 64GB Mac gives the GPU access to all 64GB. No copying, no VRAM limits.

### The Pipeline Logic
1. **CPT** teaches vocabulary and patterns (like reading a textbook)
2. **SFT** teaches behavior (like practicing Q&A with a tutor)
3. **GRPO** optimizes quality (like getting graded on your answers and improving)

Each stage builds on the previous one. You can skip CPT if your domain is already well-represented in the base model (English, code, etc.). You can skip GRPO if SFT quality is sufficient. But the full pipeline gives the best results.

---

## Citation

```bibtex
@article{onepixacademy2025grpo,
  title={Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO},
  author={Pappone, Francesco and Lazzaroni, Ruggero Marino and Califano, Federico and Gentile, Niccolò and Marras, Roberto},
  journal={arXiv preprint arXiv:2509.13081},
  year={2025},
  organization={Onepix Academy SRL}
}
```

## License

Apache 2.0

## Authors

- [Roberto Marras](https://github.com/onepixac) — Co-author of arXiv:2509.13081, CTO at Onepix Academy SRL. Architecture, implementation, and testing.
- [Claude Opus 4.6](https://anthropic.com) — Co-development and documentation
