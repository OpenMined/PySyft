"""Gemma 3 IT — Flax Inference Module

Standalone inference engine for Gemma 3 instruction-tuned models using Flax.
Module hierarchy mirrors google-deepmind/gemma so checkpoint param names map
1:1 to Flax sub-module names.

Supports: 270m, 1b, 4b, 12b, 27b.

"""

import os
import time

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import sentencepiece as spm
from flax import linen as nn


# ── Model configs ───────────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "270m": dict(
        num_layers=18,
        embed_dim=640,
        hidden_dim=2048,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        sliding_window=512,
        kaggle_handle="google/gemma-3/flax/gemma-3-270m-it",
        ckpt_subdir="gemma-3-270m-it",
    ),
    "1b": dict(
        num_layers=26,
        embed_dim=1152,
        hidden_dim=6912,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        sliding_window=512,
        kaggle_handle="google/gemma-3/flax/gemma3-1b-it",
        ckpt_subdir="gemma3-1b-it",
    ),
    "4b": dict(
        num_layers=34,
        embed_dim=2560,
        hidden_dim=10240,
        num_heads=8,
        num_kv_heads=4,
        head_dim=256,
        sliding_window=1024,
        kaggle_handle="google/gemma-3/flax/gemma3-4b-it",
        ckpt_subdir="gemma3-4b-it",
    ),
    "12b": dict(
        num_layers=48,
        embed_dim=3840,
        hidden_dim=15360,
        num_heads=16,
        num_kv_heads=8,
        head_dim=256,
        sliding_window=1024,
        kaggle_handle="google/gemma-3/flax/gemma3-12b-it",
        ckpt_subdir="gemma3-12b-it",
    ),
    "27b": dict(
        num_layers=62,
        embed_dim=5376,
        hidden_dim=21504,
        num_heads=32,
        num_kv_heads=16,
        head_dim=128,
        sliding_window=1024,
        kaggle_handle="google/gemma-3/flax/gemma3-27b-it",
        ckpt_subdir="gemma3-27b-it",
    ),
}

# ── Shared constants (identical across all Gemma 3 sizes) ─────────────────
VOCAB_SIZE = 262144
LOCAL_ROPE_BASE = 10_000
GLOBAL_ROPE_BASE = 1_000_000
K_MASK = -2.3819763e38  # Google's masking constant (≈ float32 -inf)


def _attn_types(num_layers):
    pattern = ("local",) * 5 + ("global",)
    return (pattern * ((num_layers + 5) // 6))[:num_layers]


# ── Standalone helpers ────────────────────────────────────────────────────


def apply_rope(x, positions, base_freq):
    """Rotary position embeddings (split-half rotation)."""
    half = x.shape[-1] // 2
    freq_exp = (2.0 / x.shape[-1]) * jnp.arange(half, dtype=jnp.float32)
    timescale = base_freq**freq_exp
    angles = positions[..., None, None] / timescale
    sin, cos = jnp.sin(angles), jnp.cos(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_masks(seq_len, sliding_window):
    """Causal masks — local layers also clip to a sliding window."""
    causal = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))
    window = jnp.triu(
        jnp.ones((seq_len, seq_len), dtype=jnp.bool_), k=-(sliding_window - 1)
    )
    return {
        "local": (causal & window)[None, None],
        "global": causal[None, None],
    }


def make_decode_masks(pos, sliding_window):
    """Masks for single-token decode."""
    total_len = pos + 1
    positions = jnp.arange(total_len)
    return {
        "local": (positions >= pos - sliding_window + 1)[None, None, None, :],
        "global": jnp.ones((1, 1, 1, total_len), dtype=jnp.bool_),
    }


# ── Flax modules ───────────────────────────────────────────────────────────


def _get(module, name):
    """Read a pre-loaded param without shape checking."""
    return module.variable("params", name, lambda: None).value


class Einsum(nn.Module):
    @nn.compact
    def __call__(self, equation, x):
        return jnp.einsum(equation, x, _get(self, "w"))


class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        scale = _get(self, "scale")
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-6) * (1 + scale)


class Attention(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(self, x, positions, mask, attn_type, cache=None):
        q = Einsum(name="q_einsum")("bsd,ndh->bsnh", x)
        kv = Einsum(name="kv_einsum")("bsd,ckdh->cbskh", x)
        k, v = kv[0], kv[1]

        q = RMSNorm(name="_query_norm")(q)
        k = RMSNorm(name="_key_norm")(k)

        base = LOCAL_ROPE_BASE if attn_type == "local" else GLOBAL_ROPE_BASE
        q = apply_rope(q, positions, base)
        k = apply_rope(k, positions, base)

        if cache is not None:
            cached_k, cached_v = cache
            k = jnp.concatenate([cached_k, k], axis=1)
            v = jnp.concatenate([cached_v, v], axis=1)
        new_cache = (k, v)

        q = q * (self.cfg["head_dim"] ** -0.5)

        repeats = self.cfg["num_heads"] // self.cfg["num_kv_heads"]
        k_exp = jnp.repeat(k, repeats, axis=2)
        v_exp = jnp.repeat(v, repeats, axis=2)

        logits = jnp.einsum("bsnh,btnh->bnst", q, k_exp)
        logits = jnp.where(mask, logits, K_MASK)
        weights = jax.nn.softmax(logits, axis=-1)

        out = jnp.einsum("bnst,btnh->bsnh", weights, v_exp)
        return Einsum(name="attn_vec_einsum")("bsnh,nhd->bsd", out), new_cache


class FeedForward(nn.Module):
    @nn.compact
    def __call__(self, x):
        gate = Einsum(name="gating_einsum")("bsf,nhf->bsnh", x)
        h = jax.nn.gelu(gate[:, :, 0, :]) * gate[:, :, 1, :]
        return Einsum(name="linear")("bsh,hf->bsf", h)


class Block(nn.Module):
    cfg: dict
    attn_type: str = "local"

    @nn.compact
    def __call__(self, x, positions, mask, cache=None):
        h = RMSNorm(name="pre_attention_norm")(x)
        h, new_cache = Attention(cfg=self.cfg, name="attn")(
            h, positions, mask, self.attn_type, cache
        )
        h = RMSNorm(name="post_attention_norm")(h)
        x = x + h
        h = RMSNorm(name="pre_ffw_norm")(x)
        h = FeedForward(name="mlp")(h)
        h = RMSNorm(name="post_ffw_norm")(h)
        return x + h, new_cache


class Embedder(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(self, token_ids):
        table = _get(self, "input_embedding")
        return table[token_ids] * jnp.sqrt(float(self.cfg["embed_dim"])), table


class Transformer(nn.Module):
    cfg: dict

    @nn.compact
    def __call__(self, tokens, cache=None):
        sliding_window = self.cfg["sliding_window"]
        num_layers = self.cfg["num_layers"]
        attn_types = _attn_types(num_layers)

        x, embed_table = Embedder(cfg=self.cfg, name="embedder")(tokens)

        if cache is None:
            seq_len = tokens.shape[1]
            positions = jnp.arange(seq_len)[None, :]
            masks = make_masks(seq_len, sliding_window)
        else:
            cache_len = cache["layer_0"][0].shape[1]
            positions = jnp.array([[cache_len]])
            masks = make_decode_masks(cache_len, sliding_window)

        new_cache = {}
        for i in range(num_layers):
            layer_cache = cache[f"layer_{i}"] if cache is not None else None
            x, layer_new_cache = Block(
                cfg=self.cfg, attn_type=attn_types[i], name=f"layer_{i}"
            )(x, positions, masks[attn_types[i]], layer_cache)
            new_cache[f"layer_{i}"] = layer_new_cache

        x = RMSNorm(name="final_norm")(x)
        logits = x @ embed_table.T
        return logits, new_cache


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


def setup(size, weights_dir):
    """Configure model, load weights and tokenizer.

    Returns (model, tokenizer, params).
    """
    cfg = MODEL_CONFIGS[size]
    params = load_params(weights_dir, cfg)
    model = Transformer(cfg=cfg)
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
