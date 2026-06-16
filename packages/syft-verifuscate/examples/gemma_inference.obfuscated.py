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


# THIS COMMENT WAS OBFUSCATED
# THIS COMMENT WAS OBFUSCATED
░v0 = dict(
    ░v5=■,
    embed_dim=■,
    hidden_dim=■,
    num_heads=■,
    num_kv_heads=■,
    head_dim=■,
    ░v19=■,
    kaggle_handle="■",
    ckpt_subdir="■",
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
░v1 = ■
░v2 = ■
░v3 = ■
░v4 = -■  # THIS COMMENT WAS OBFUSCATED


def ░fn1(░v5):
    ░v6 = ("■",) * ■ + ("■",)
    return (░v6 * ((░v5 + ■) // ■))[:░v5]


# ── Standalone helpers ────────────────────────────────────────────────────


def ░fn0(░v7, ░v8, ░v9):
    "■"
    ░v10 = shape_of(░v7)[-■] // ■
    ░v11 = (■ / shape_of(░v7)[-■]) * jnp.arange(░v10, dtype=jnp.float32)
    ░v12 = ░v9**░v11
    ░v13 = ░v8[..., None, None] / ░v12
    ░v14, ░v15 = jnp.sin(░v13), jnp.cos(░v13)
    ░v16, ░v17 = ░v7[..., :░v10], ░v7[..., ░v10:]
    return jnp.concatenate([░v16 * ░v15 - ░v17 * ░v14, ░v17 * ░v15 + ░v16 * ░v14], axis=-■)


def ░fn2(░v18, ░v19):
    "■"
    ░v20 = jnp.tril(jnp.ones((░v18, ░v18), dtype=jnp.bool_))
    ░v21 = jnp.triu(
        jnp.ones((░v18, ░v18), dtype=jnp.bool_), ░v32=-(░v19 - ■)
    )
    return {
        "■": (░v20 & ░v21)[None, None],
        "■": ░v20[None, None],
    }


def ░fn3(░v22, ░v19):
    "■"
    ░v23 = ░v22 + ■
    ░v8 = jnp.arange(░v23)
    return {
        "■": (░v8 >= ░v22 - ░v19 + ■)[None, None, None, :],
        "■": jnp.ones((■, ■, ■, ░v23), dtype=jnp.bool_),
    }


# ── Flax modules ───────────────────────────────────────────────────────────


def _get(module, name):
    """Read a pre-loaded param without shape checking."""
    return module.variable("params", name, lambda: None).value


def transpose(a):
    """Visible wrapper: matrix transpose (a value transform, not a metadata read)."""
    return a.T


def shape_of(x):
    """Visible wrapper: read an array's shape — an attribute read on a value, not allowed in the hidden region."""
    return x.shape


def append_to(lst, item):
    """Visible wrapper: append to a Python list (a named method on a value)."""
    lst.append(item)
    return lst


class ░Cls0(nn.Module):
    def setup(self):
        self.░a20 = _get(self, "■")

    def __call__(self, ░v24, ░v7):
        return jnp.einsum(░v24, ░v7, self.░a20)


class ░Cls1(nn.Module):
    def setup(self):
        self.░a19 = _get(self, "■")

    def __call__(self, ░v7):
        ░v25 = jnp.mean(jnp.square(░v7), axis=-■, keepdims=True)
        return ░v7 * jax.lax.rsqrt(░v25 + ■) * (■ + self.░a19)


class ░Cls2(nn.Module):
    ░v26: dict

    def setup(self):
        self.░a18 = ░Cls0()
        self.░a10 = ░Cls0()
        self.░a1 = ░Cls1()
        self.░a0 = ░Cls1()
        self.░a4 = ░Cls0()

    def __call__(self, ░v7, ░v8, ░v27, ░v28, ░v29=None):
        ░v30 = self.░a18("■", ░v7)
        ░v31 = self.░a10("■", ░v7)
        ░v32, ░v33 = ░v31[■], ░v31[■]

        ░v30 = self.░a1(░v30)
        ░v32 = self.░a0(░v32)

        ░v34 = ░v2 if ░v28 == "■" else ░v3
        ░v30 = ░fn0(░v30, ░v8, ░v34)
        ░v32 = ░fn0(░v32, ░v8, ░v34)

        if ░v29 is not None:
            ░v35, ░v36 = ░v29
            ░v32 = jnp.concatenate([░v35, ░v32], axis=■)
            ░v33 = jnp.concatenate([░v36, ░v33], axis=■)
        ░v37 = (░v32, ░v33)

        ░v30 = ░v30 * (self.░a5["■"] ** -■)

        ░v38 = self.░a5["■"] // self.░a5["■"]
        ░v39 = jnp.repeat(░v32, ░v38, axis=■)
        ░v40 = jnp.repeat(░v33, ░v38, axis=■)

        ░v41 = jnp.einsum("■", ░v30, ░v39)
        ░v41 = jnp.where(░v27, ░v41, ░v4)
        ░v42 = jax.nn.softmax(░v41, axis=-■)

        ░v43 = jnp.einsum("■", ░v42, ░v40)
        return self.░a4("■", ░v43), ░v37


class ░Cls3(nn.Module):
    def setup(self):
        self.░a8 = ░Cls0()
        self.░a12 = ░Cls0()

    def __call__(self, ░v7):
        ░v44 = self.░a8("■", ░v7)
        ░v45 = jax.nn.gelu(░v44[:, :, ■, :]) * ░v44[:, :, ■, :]
        return self.░a12("■", ░v45)


class ░Cls5(nn.Module):
    ░v26: dict
    ░v28: str = "■"

    def setup(self):
        self.░a16 = ░Cls1()
        self.░a2 = ░Cls2(░v26=self.░a5)
        self.░a14 = ░Cls1()
        self.░a17 = ░Cls1()
        self.░a13 = ░Cls3()
        self.░a15 = ░Cls1()

    def __call__(self, ░v7, ░v8, ░v27, ░v29=None):
        ░v45 = self.░a16(░v7)
        ░v45, ░v37 = self.░a2(░v45, ░v8, ░v27, self.░a3, ░v29)
        ░v45 = self.░a14(░v45)
        ░v7 = ░v7 + ░v45
        ░v45 = self.░a17(░v7)
        ░v45 = self.░a13(░v45)
        ░v45 = self.░a15(░v45)
        return ░v7 + ░v45, ░v37


class ░Cls4(nn.Module):
    ░v26: dict

    def setup(self):
        self.░a9 = _get(self, "■")

    def __call__(self, ░v46):
        ░v47 = self.░a9
        return ░v47[░v46] * jnp.sqrt(float(self.░a5["■"])), ░v47


class Transformer(nn.Module):
    ░v26: dict

    def setup(self):
        ░v5 = self.░a5["■"]
        ░v48 = ░fn1(░v5)
        self.░a6 = ░Cls4(░v26=self.░a5)
        self.░a11 = [
            ░Cls5(░v26=self.░a5, ░v28=░v48[░v49]) for ░v49 in range(░v5)
        ]
        self.░a7 = ░Cls1()

    def __call__(self, ░v50, ░v29=None):
        ░v19 = self.░a5["■"]
        ░v5 = self.░a5["■"]
        ░v48 = ░fn1(░v5)

        ░v7, ░v51 = self.░a6(░v50)

        if ░v29 is None:
            ░v18 = shape_of(░v50)[■]
            ░v8 = jnp.arange(░v18)[None, :]
            ░v52 = ░fn2(░v18, ░v19)
        else:
            ░v53 = shape_of(░v29[■][■])[■]
            ░v8 = jnp.array([[░v53]])
            ░v52 = ░fn3(░v53, ░v19)

        ░v37 = []
        for ░v49 in range(░v5):
            ░v54 = ░v29[░v49] if ░v29 is not None else None
            ░v55 = self.░a11[░v49]
            ░v7, ░v56 = ░v55(░v7, ░v8, ░v52[░v48[░v49]], ░v54)
            ░v37 = append_to(░v37, ░v56)

        ░v7 = self.░a7(░v7)
        ░v41 = ░v7 @ transpose(░v51)
        return ░v41, ░v37


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


def setup(weights_dir):
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
