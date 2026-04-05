#!/usr/bin/env python3
"""
GRPO Training on MLX — Semantic Reward Optimization
Based on arXiv:2509.13081

Generator-Evaluator architecture with multi-criteria reward and adaptive temperature.
IMPORTANT: Run on a FUSED model (not on raw LoRA adapter) — see README.md Known Issues #1.
"""

import json
import time
import sys
import numpy as np
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx.utils import tree_flatten

# === CONFIGURATION ===
MODEL_PATH = "models/fused-sft"
DATASET_PATH = "data/sft/train.jsonl"
RESULTS_PATH = "grpo_results.json"
LOG_PATH = "grpo_train.log"
GRPO_ADAPTER_DIR = Path("adapters/stage3-grpo")

CONFIG = {
    "n_candidates": 4,
    "max_tokens": 256,
    "learning_rate": 1e-6,
    "n_iters": 30,
    "batch_size": 3,
    "lora_layers": 16,
    "min_response_length": 80,
    "max_attempts": 5,
}


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def save_lora_weights(model, path, label=""):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.savez(str(path / "adapters.npz"), **weights)
    log(f"Saved LoRA weights to {path} ({label}, {len(weights)} tensors)")


def load_reference_data(min_length=80):
    data = []
    with open(DATASET_PATH, "r") as f:
        for line in f:
            entry = json.loads(line)
            msgs = entry["messages"]
            if len(msgs) == 3 and len(msgs[2]["content"]) >= min_length:
                data.append({
                    "system": msgs[0]["content"],
                    "prompt": msgs[1]["content"],
                    "reference": msgs[2]["content"]
                })
    return data


def compute_reward(encoder, response, reference):
    if not response.strip():
        return 0.0
    embs = encoder.encode([response, reference], normalize_embeddings=True)
    semantic = float(np.dot(embs[0], embs[1]))
    ref_len = len(reference)
    resp_len = len(response)
    len_ratio = min(resp_len, ref_len) / max(resp_len, ref_len) if max(resp_len, ref_len) > 0 else 0
    brevity = 1.0 - min(abs(resp_len - ref_len) / max(ref_len, 1), 1.0)
    return 0.7 * max(semantic, 0) + 0.2 * len_ratio + 0.1 * brevity


def generate_response(model, tokenizer, system, prompt, max_tokens, temperature):
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False  # Disable thinking mode if supported (safe to keep for all models)
    )
    sampler = make_sampler(temp=temperature)
    return generate(model, tokenizer, prompt=text, max_tokens=max_tokens, sampler=sampler)


def grpo_loss_fn(model, tokenizer, prompt_text, candidates, rewards):
    mean_r = np.mean(rewards)
    advantages = [(r - mean_r) for r in rewards]
    total_loss = mx.array(0.0)
    for candidate, advantage in zip(candidates, advantages):
        full_text = prompt_text + candidate
        tokens = mx.array(tokenizer.encode(full_text))[None, :]
        logits = model(tokens[:, :-1])
        targets = tokens[:, 1:]
        log_probs = -nn.losses.cross_entropy(logits, targets, reduction="mean")
        total_loss = total_loss - log_probs * advantage
    return total_loss / len(candidates)


def run_grpo(config=None):
    if config is None:
        config = CONFIG.copy()

    log(f"Starting GRPO Training")

    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    log("Encoder loaded")

    model, tokenizer = load(MODEL_PATH)
    log(f"Model loaded from {MODEL_PATH} (fused SFT)")

    model.freeze()
    lora_config = {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0}
    linear_to_lora_layers(model, config["lora_layers"], lora_config)
    model.train()
    log(f"LoRA applied ({config['lora_layers']} layers)")

    optimizer = optim.Adam(learning_rate=config["learning_rate"])
    data = load_reference_data(config.get("min_response_length", 80))
    loss_and_grad = nn.value_and_grad(model, lambda m, p, c, r: grpo_loss_fn(m, tokenizer, p, c, r))

    log(f"Data: {len(data)} prompts")

    results = []
    best_avg_reward = 0.0
    no_improve = 0
    current_temp = 0.70

    import random
    random.seed(42)

    for iteration in range(config["n_iters"]):
        start = time.time()
        batch = random.sample(data, min(config["batch_size"], len(data)))
        batch_rewards = []
        batch_losses = []

        for item in batch:
            messages = [
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["prompt"]},
            ]
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode if supported (safe to keep for all models)
            )

            candidates = []
            rewards = []
            for _ in range(config["n_candidates"]):
                resp = generate_response(
                    model, tokenizer, item["system"], item["prompt"],
                    config["max_tokens"], current_temp
                )
                r = compute_reward(encoder, resp, item["reference"])
                candidates.append(resp)
                rewards.append(r)

            batch_rewards.extend(rewards)

            loss, grads = loss_and_grad(model, prompt_text, candidates, rewards)
            optimizer.update(model, grads)
            # Evaluate parameters and optimizer state
            params = model.parameters()
            opt_state = optimizer.state
            mx.eval(mx.utils.tree_flatten(params))
            mx.eval(mx.utils.tree_flatten(opt_state))
            batch_losses.append(loss.item())

        elapsed = time.time() - start
        avg_r = np.mean(batch_rewards)
        max_r = np.max(batch_rewards)
        avg_l = np.mean(batch_losses)

        results.append({
            "iteration": iteration + 1, "avg_reward": avg_r, "max_reward": max_r,
            "loss": avg_l, "temperature": current_temp, "time": elapsed
        })
        log(f"Iter {iteration+1}/{config['n_iters']}: reward={avg_r:.3f} (max={max_r:.3f}), "
            f"loss={avg_l:.4f}, temp={current_temp:.2f}, {elapsed:.1f}s")

        if avg_r > best_avg_reward + 0.01:
            best_avg_reward = avg_r
            no_improve = 0
            save_lora_weights(model, GRPO_ADAPTER_DIR / "best", f"best reward={avg_r:.3f}")
        else:
            no_improve += 1
        if no_improve >= 5 and iteration >= 10:
            log("Early stop: no improvement")
            break

    save_lora_weights(model, GRPO_ADAPTER_DIR / "final", "final")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"results": results, "best_reward": best_avg_reward}, f, indent=2)

    log(f"GRPO Complete. Best reward: {best_avg_reward:.3f}")
    return best_avg_reward


if __name__ == "__main__":
    for attempt in range(CONFIG["max_attempts"]):
        log(f"=== GRPO Attempt {attempt + 1} ===")
        try:
            reward = run_grpo()
            log(f"GRPO succeeded with reward {reward:.3f}")
            break
        except Exception as e:
            log(f"ERROR attempt {attempt + 1}: {e}")
            if attempt < CONFIG["max_attempts"] - 1:
                CONFIG["lora_layers"] = max(4, CONFIG["lora_layers"] - 2)
    log("GRPO Experiment Complete")
