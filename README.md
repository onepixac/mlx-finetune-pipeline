# MLX Fine-Tuning Pipeline: CPT → SFT → GRPO on Apple Silicon

A complete, production-ready pipeline for fine-tuning language models on Apple Silicon Macs using MLX. Includes Continued Pre-Training (CPT), Supervised Fine-Tuning (SFT), and Group Relative Policy Optimization (GRPO) with semantic reward modeling.

Based on the paper: ["Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO"](https://arxiv.org/abs/2509.13081)

## Why Apple Silicon?

| | Apple Silicon (MLX) | NVIDIA GPU (CUDA) |
|---|---|---|
| **Cost** | Mac Mini M4 Pro ~$2,000 one-time | A100 ~$2-3/hour cloud |
| **Memory** | 64-192GB unified (shared CPU/GPU) | 24-80GB VRAM |
| **Setup** | `pip install mlx-lm` | CUDA drivers, Docker, etc. |
| **Power** | ~50W | ~300W per GPU |
| **Noise** | Silent | Data center |

**Unified memory is the key advantage.** A Mac with 64GB RAM can fine-tune models that would require 2x A100s on NVIDIA, because CPU and GPU share the same memory pool with no transfer overhead.

## What Models Can You Fine-Tune?

### Memory Requirements (4-bit quantized, LoRA training)

| Model | Parameters | 4-bit Size | Training RAM | Mac Needed |
|---|---|---|---|---|
| Gemma 4 E2B | 2B | ~1.5 GB | ~8 GB | Any M1+ (16GB) |
| Qwen3.5-4B | 4B | ~2.5 GB | ~12 GB | M1/M2/M3/M4 (16GB) |
| Gemma 4 E4B | 4B active (12B total) | ~7 GB | ~15 GB | M1+ (24GB) |
| Qwen3.5-9B | 9B | ~5.6 GB | ~14 GB | M1+ (24GB) |
| Gemma 4 26B-A4B | 4B active (26B total) | ~15.6 GB | ~22 GB | M1+ (32GB) |
| Qwen3.5-30B-A3B | 3B active (30B total) | ~17 GB | ~25 GB | M2+ (32GB) |
| Llama 3.3-70B | 70B | ~40 GB | ~50 GB | M2+ (64GB) |
| Qwen3.5-32B | 32B | ~18 GB | ~28 GB | M2+ (48GB) |

**Rule of thumb:** Model size in 4-bit + ~6-10 GB for training overhead = minimum RAM needed.

## The Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: CPT (Continued Pre-Training)                       │
│  Purpose: Familiarize model with domain-specific text        │
│  Input: Raw text chunks {"text": "..."}                      │
│  Output: Domain-adapted LoRA adapter                         │
│  Duration: 1-3 hours (1000-3000 iterations)                  │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Stage 2: SFT (Supervised Fine-Tuning)                       │
│  Purpose: Teach the model to follow instructions             │
│  Input: Chat pairs {"messages": [system, user, assistant]}   │
│  Output: Instruction-tuned LoRA adapter                      │
│  Duration: 12-72 hours depending on dataset size             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Stage 3: GRPO (Semantic Reward Optimization)                │
│  Purpose: Optimize response quality with reward model        │
│  Input: Prompts + semantic evaluator                         │
│  Output: Quality-optimized LoRA adapter                      │
│  Duration: 1-2 hours (30 iterations)                         │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install mlx-lm sentence-transformers

# For Gemma 4 support: install from main branch
pip install 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@main'
```

### Download a Model

```bash
# Example: Gemma 4 26B-A4B (15.6 GB, fits on 32GB+ Mac)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('mlx-community/gemma-4-26b-a4b-it-4bit', local_dir='models/gemma4-26b-4bit')
"
```

### Stage 1: CPT

```bash
python -m mlx_lm lora -c config_cpt.yaml
```

### Stage 2: SFT

```bash
python -m mlx_lm lora -c config_sft.yaml
```

### Stage 3: GRPO

```bash
# IMPORTANT: Fuse SFT adapter first (see Known Issues #1)
python -m mlx_lm fuse \
  --model models/your-base-model \
  --adapter-path adapters/stage2-sft \
  --save-path models/fused-sft

# Then run GRPO on fused model
python grpo_train.py
```

## Configuration

### CPT Config (`config_cpt.yaml`)

```yaml
model: models/your-base-model
data: data/cpt            # {"text": "raw domain text..."}
adapter_path: adapters/stage1-cpt
train: true
fine_tune_type: lora
mask_prompt: false         # CPT: don't mask anything

num_layers: 16             # LoRA layers
iters: 3000
batch_size: 1
learning_rate: 5e-6        # Low LR for CPT
steps_per_report: 100
steps_per_eval: 500
val_batches: 10
save_every: 500            # Checkpoint every 500 iters
max_seq_length: 1024
grad_checkpoint: true
seed: 42
```

### SFT Config (`config_sft.yaml`)

```yaml
model: models/your-base-model
data: data/sft            # {"messages": [system, user, assistant]}
adapter_path: adapters/stage2-sft
resume_adapter_file: adapters/stage1-cpt/best_checkpoint.safetensors
train: true
fine_tune_type: lora
mask_prompt: true          # SFT: mask system+user, train on assistant only

num_layers: 16
iters: 170000              # dataset_size * num_epochs
batch_size: 1
learning_rate: 1e-5
steps_per_report: 500
steps_per_eval: 10000
val_batches: 25
save_every: 5000           # Protect against OOM crashes
max_seq_length: 512
grad_checkpoint: true
seed: 42
```

## Dataset Format

### CPT Data
```json
{"text": "Your raw domain text goes here. The model reads this to familiarize itself with the vocabulary and patterns of your target domain."}
```

### SFT Data
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert in X. Always respond in Y format."},
    {"role": "user", "content": "User's question in natural language"},
    {"role": "assistant", "content": "Model's response — this is what the model learns to generate"}
  ]
}
```

### Dataset Best Practices

**DO:**
- Use specific, natural user prompts that match real usage
- Keep assistant responses in the target style/language
- Balance your dataset: ~40% conversation, ~40% knowledge, ~20% corrections
- Deduplicate by assistant content
- Verify a sample manually before training

**DON'T:**
- Use generic prompts like "Tell me about X" or "Read this text"
- Put the same text in both user and assistant
- Create "read me this passage" → passage pairs (model learns to parrot, not converse)
- Skip quality verification
- Train on AI-generated responses without verification

### The Critical Lesson: Format Matters More Than Size

A 12,000-pair dataset in the wrong format produces a model that invents words. A 5,000-pair dataset in the right format produces a model that responds correctly.

**Wrong format** (model learns to recite):
```json
{"user": "Read me paragraph 3 of chapter 1", "assistant": "The actual paragraph text..."}
```

**Right format** (model learns to converse):
```json
{"user": "What happened when the princess looked at the sky?", "assistant": "The actual paragraph text..."}
```

Same source material, same response — different user prompt. The model learns the pattern: when someone asks X, respond with Y.

## Known Issues & Fixes

### 1. GRPO Fails with "Can't convert LoRALinear to LoRA"

**Problem:** GRPO tries to apply LoRA on top of an SFT adapter that already has LoRA layers.

**Solution:** Fuse the SFT adapter into the base model before running GRPO:

```bash
# Fuse SFT adapter
python -m mlx_lm fuse \
  --model models/your-base-model \
  --adapter-path adapters/stage2-sft \
  --save-path models/fused-sft-model

# Run GRPO on fused model (no adapter_path needed)
# In grpo_train.py, set MODEL_PATH to the fused model
```

### 2. Gemma 4 Won't Load in mlx-lm

**Problem:** Released mlx-lm versions (≤0.31.2) don't fully support Gemma 4's MoE architecture. You get "510 parameters not in model" error.

**Solution:** Install mlx-lm from the main branch:

```bash
pip install 'mlx-lm @ git+https://github.com/ml-explore/mlx-lm.git@main'
```

### 3. OOM (Out of Memory) During Training

**Problem:** Training crashes after hours with "Insufficient Memory" error, losing all progress.

**Solution:**
- Set `save_every: 5000` in config to checkpoint regularly
- Don't run other GPU-intensive tasks during training (embedding models, scraping, etc.)
- Reduce `max_seq_length` if needed (512 is enough for short responses)
- Use `grad_checkpoint: true` to trade speed for memory

### 4. CPT Overfitting

**Problem:** CPT val loss improves then starts rising (overfitting on small text corpus).

**Solution:** Monitor val loss every 500 iterations. Use the checkpoint with the lowest val loss, not the final one. Typical best checkpoint is at 500-1500 iterations.

### 5. Resume Training After Crash

**Problem:** Training crashed and you want to resume from the last checkpoint.

**Solution:**
```bash
python -m mlx_lm lora -c config_sft.yaml \
  --resume-adapter-file adapters/stage2-sft/best_checkpoint.safetensors \
  --iters REMAINING_ITERATIONS
```

Note: The optimizer state is not saved, so the model needs a few hundred iterations to "warm up" again. The val loss will be temporarily higher than the checkpoint's val loss.

## GRPO: Semantic Reward Optimization

GRPO uses a Generator-Evaluator architecture inspired by [Anthropic's harness design](https://www.anthropic.com/engineering/harness-design):

```
For each training prompt:
  1. Generator: produce N candidate responses
  2. Evaluator: score each candidate on semantic similarity,
     response length, and brevity match
  3. Reinforce: update weights to favor high-scoring candidates
```

### GRPO Config

```python
CONFIG = {
    "n_candidates": 4,       # Candidates per prompt
    "max_tokens": 256,       # Max response length
    "learning_rate": 1e-6,   # Very low LR for GRPO
    "n_iters": 30,           # GRPO iterations
    "lora_layers": 16,
}
```

### Critical: Disable Thinking Mode

For models with thinking mode (Qwen3.5, etc.), you MUST set `enable_thinking=False` in `apply_chat_template`. With thinking mode: reward ~0.27. Without: reward ~0.92.

## Full Pipeline Script

```bash
#!/bin/bash
# run_pipeline.sh — Full CPT → SFT → GRPO pipeline

set -e
source venv/bin/activate

echo "=== Stage 1: CPT ==="
python -m mlx_lm lora -c config_cpt.yaml

echo "=== Stage 2: SFT ==="
python -m mlx_lm lora -c config_sft.yaml

echo "=== Fuse SFT for GRPO ==="
python -m mlx_lm fuse \
  --model models/your-base-model \
  --adapter-path adapters/stage2-sft \
  --save-path models/fused-sft

echo "=== Stage 3: GRPO ==="
python grpo_train.py

echo "=== Pipeline complete ==="
```

## Hardware Tested

| Mac | RAM | Models Tested | Notes |
|---|---|---|---|
| Mac Mini M4 Pro | 64 GB | Qwen3.5-9B, Gemma 4 26B-A4B | Training ~0.5-0.6 it/sec |
| MacBook Pro M3 Max | 48 GB | Qwen3.5-9B | Training ~0.4 it/sec |
| Mac Studio M2 Ultra | 192 GB | Up to 70B models | Untested but should work |

## Citation

If you use this pipeline, please cite:

```bibtex
@article{pappone2025grpo,
  title={Shaping Explanations: Semantic Reward Modeling with Encoder-Only Transformers for GRPO},
  author={Pappone, Lazzaroni, Califano, Gentile, Marras},
  journal={arXiv preprint arXiv:2509.13081},
  year={2025}
}
```

## License

Apache 2.0
