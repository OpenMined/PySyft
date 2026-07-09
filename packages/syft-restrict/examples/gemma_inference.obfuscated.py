"""Gemma 3 IT — Flax Inference Module

Standalone inference engine for Gemma 3 instruction-tuned models using Flax.
Module hierarchy mirrors google-deepmind/gemma so checkpoint param names map
1:1 to Flax sub-module names.

Supports: 270m, 1b, 4b, 12b, 27b.

Adapted from: github.com/anthropics/beach-notebooks/gemma/local_PoC.py
"""

import os
import time

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import sentencepiece as spm
from flax import linen as nn


# ── Model config ─────────────────────────────────────────────────────────────
# Active model config. Comment/uncomment to switch sizes.
░v0 = dict(
    ░v1=■,
    ░v2=■,
    ░v3=■,
    ░v4=■,
    ░v5=■,
    ░v6=■,
    ░v7=■,
    ░v8="■",
    ░v9="■",
)

# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED

# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED

# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED

# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED

# THIS COMMENT WAS OBFUSCATED
░v10 = ■
░v11 = ■
░v12 = ■
░v13 = -■  # THIS COMMENT WAS OBFUSCATED


def ░fn0(░v1):
    ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
    ■■■■■■■■


# ── Standalone helpers ────────────────────────────────────────────────────


def ░fn1(░v14, ░v15, ░v16):
    ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
    ■■■■■■■■
    ■■■■■■■■
    ■■■■■■■■
    ■■■■■■■■
    ■■■■■■■■
    ■■■■■■■■
    ■■■■■■■■


def ░fn2(░v17, ░v7):
    ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
    ■■■■■■■■
    ■■■■■■■■
        ■■■■■■■■
    ■■■■■■■■
    ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
    ■■■■■■■■


def ░fn3(░v18, ░v7):
    ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
    ■■■■■■■■
    ■■■■■■■■
    ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
    ■■■■■■■■


# ── Flax modules ───────────────────────────────────────────────────────────


def _get(module, name):
    """Read a pre-loaded param without shape checking."""
    return module.variable("params", name, lambda: None).value


def shape_of(x):
    """Visible wrapper: read an array's shape — an attribute read on a value, not allowed in the hidden region."""
    return x.shape


def append_to(lst, item):
    """Visible wrapper: append to a Python list (a named method on a value)."""
    lst.append(item)
    return lst


class ░Cls0(nn.Module):
    def setup(self):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details

    def __call__(self, ░v19, ░v14):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details


class ░Cls1(nn.Module):
    def setup(self):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details

    def __call__(self, ░v14):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■


class ░Cls2(nn.Module):
    ░v20: dict

    def setup(self):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■

    def __call__(self, ░v14, ░v15, ░v21, ░v22, ░v23=None):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■
        ■■■■■■■■

        ■■■■■■■■
        ■■■■■■■■

        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■

        ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■
        ■■■■■■■■

        ■■■■■■■■

        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■

        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■

        ■■■■■■■■
        ■■■■■■■■


class ░Cls3(nn.Module):
    def setup(self):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■

    def __call__(self, ░v14):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■
        ■■■■■■■■


class ░Cls4(nn.Module):
    ░v20: dict
    ░v22: str = "■"

    def setup(self):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■

    def __call__(self, ░v14, ░v15, ░v21, ░v23=None):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■


class ░Cls5(nn.Module):
    ░v20: dict

    def setup(self):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details

    def __call__(self, ░v24):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■


class ░Cls6(nn.Module):
    ░v20: dict

    def setup(self):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■
            ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■

    def __call__(self, ░v25, ░v23=None):
        ■■■■■■■■  # hidden/obfuscated lines can only execute restricted python, see restrict docs for more details
        ■■■■■■■■
        ■■■■■■■■

        ■■■■■■■■

        ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■
        ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■

        ■■■■■■■■
        ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■
            ■■■■■■■■

        ■■■■■■■■
        ■■■■■■■■
        ■■■■■■■■


# ── Weight loading ─────────────────────────────────────────────────────────


def nestify(flat):
    """Convert Orbax flat dict to nested dict for Flax."""
    nested = {}
    for flat_key, param_dict in flat.items():
        parts = flat_key.split("/")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = param_dict
    return nested


def load_params(weights_dir, cfg):
    """Load Orbax checkpoint and return Flax-compatible params dict."""
    ckpt_path = os.path.join(weights_dir, cfg["ckpt_subdir"])
    raw = ocp.PyTreeCheckpointer().restore(ckpt_path)
    return {"params": nestify(raw)["transformer"]}


# ── Setup (convenience entry point) ───────────────────────────────────────


def setup_model(weights_dir):
    """Configure model, load weights and tokenizer.

    Returns (model, tokenizer, params).
    """
    params = load_params(weights_dir, CONFIG)
    model = Transformer(cfg=CONFIG)
    sp = load_tokenizer(weights_dir)
    return model, sp, params


# ── Tokenizer + generation ─────────────────────────────────────────────────


def load_tokenizer(weights_dir):
    """Load SentencePiece tokenizer from weights directory."""
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.join(weights_dir, "tokenizer.model"))
    return sp


def format_chat(prompt):
    """Wrap prompt in Gemma's chat template."""
    return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"


def sample_token(logits, temperature=0.8, top_k=40):
    """Temperature-scaled top-k sampling. Greedy when temperature=0."""
    if temperature == 0:
        return int(jnp.argmax(logits))
    logits = logits / temperature
    top_k_logits, top_k_ids = jax.lax.top_k(logits, top_k)
    probs = jax.nn.softmax(top_k_logits)
    idx = jax.random.categorical(
        jax.random.PRNGKey(int(jnp.sum(logits) * 1e6) % 2**31),
        jnp.log(probs),
    )
    return int(top_k_ids[idx])


def generate(model, params, sp, prompt, max_new_tokens=200, temperature=0.8, top_k=40):
    """Autoregressive generation with KV cache and chat template.

    Returns (response_text, stats_dict).
    """
    chat_input = format_chat(prompt)
    token_ids = [sp.bos_id()] + sp.EncodeAsIds(chat_input)
    prompt_tokens = jnp.array([token_ids], dtype=jnp.int32)
    prompt_text = sp.Decode(token_ids)
    generated_ids = list(token_ids)

    t_prefill = time.time()
    logits, cache = model.apply(params, prompt_tokens)
    ttft = time.time() - t_prefill

    t_decode = time.time()
    decode_tokens = 0

    for _ in range(max_new_tokens):
        next_id = sample_token(logits[0, -1], temperature, top_k)
        if next_id == sp.eos_id():
            break

        sp.Decode(generated_ids)
        generated_ids.append(next_id)
        new_text = sp.Decode(generated_ids)

        response_so_far = new_text[len(prompt_text) :]
        if "<end_of_turn>" in response_so_far:
            break

        decode_tokens += 1
        logits, cache = model.apply(
            params, jnp.array([[next_id]], dtype=jnp.int32), cache=cache
        )

    decode_elapsed = time.time() - t_decode
    decode_tps = decode_tokens / decode_elapsed if decode_elapsed > 0 else 0

    full = sp.Decode(generated_ids)
    response_start = full.find("<start_of_turn>model\n")
    if response_start != -1:
        response = full[response_start + len("<start_of_turn>model\n") :]
        response = response.replace("<end_of_turn>", "").strip()
    else:
        response = full

    stats = {
        "ttft": ttft,
        "decode_tps": decode_tps,
        "decode_tokens": decode_tokens,
        "decode_elapsed": decode_elapsed,
        "prompt_tokens": len(token_ids),
    }
    return response, stats
