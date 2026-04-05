#!/bin/bash
# Full CPT → SFT → GRPO Pipeline for MLX on Apple Silicon
# Usage: ./run_pipeline.sh

set -e

echo "============================================"
echo "  MLX Fine-Tuning Pipeline"
echo "  CPT → SFT → GRPO on Apple Silicon"
echo "============================================"

# Activate virtual environment
source venv/bin/activate

# Check model exists
if [ ! -d "models/your-base-model" ]; then
    echo "ERROR: Model not found at models/your-base-model"
    echo "Download a model first. Example:"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('mlx-community/gemma-4-26b-a4b-it-4bit', local_dir='models/your-base-model')\""
    exit 1
fi

# Check data exists
if [ ! -f "data/cpt/train.jsonl" ]; then
    echo "ERROR: CPT data not found at data/cpt/train.jsonl"
    exit 1
fi
if [ ! -f "data/sft/train.jsonl" ]; then
    echo "ERROR: SFT data not found at data/sft/train.jsonl"
    exit 1
fi

echo ""
echo "=== Stage 1: CPT (Continued Pre-Training) ==="
echo "This familiarizes the model with your domain text."
echo ""
python -m mlx_lm lora -c config_cpt.yaml
echo "CPT complete."

# Find best CPT checkpoint (lowest val loss)
echo ""
echo "=== Finding best CPT checkpoint ==="
BEST_CPT=$(ls -t adapters/stage1-cpt/0*_adapters.safetensors 2>/dev/null | head -1)
if [ -z "$BEST_CPT" ]; then
    BEST_CPT="adapters/stage1-cpt/adapters.safetensors"
fi
echo "Using CPT checkpoint: $BEST_CPT"

echo ""
echo "=== Stage 2: SFT (Supervised Fine-Tuning) ==="
echo "This teaches the model to follow your instructions."
echo ""
python -m mlx_lm lora -c config_sft.yaml --resume-adapter-file "$BEST_CPT"
echo "SFT complete."

echo ""
echo "=== Fusing SFT adapter into base model ==="
python -m mlx_lm fuse \
    --model models/your-base-model \
    --adapter-path adapters/stage2-sft \
    --save-path models/fused-sft
echo "Fuse complete."

echo ""
echo "=== Stage 3: GRPO (Semantic Reward Optimization) ==="
echo "This optimizes response quality using reward signals."
echo ""
python grpo_train.py
echo "GRPO complete."

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Final model: models/fused-sft + adapters/stage3-grpo"
echo "============================================"
echo ""
echo "To fuse the final model:"
echo "  python -m mlx_lm fuse --model models/fused-sft --adapter-path adapters/stage3-grpo/best --save-path models/final"
