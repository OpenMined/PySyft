"""Gemma 3 IT — Flax Inference Module (syft-restrict compliant, optimized)

Same model and same public API as gemma_inference_restrict.py — MODEL_CONFIGS,
setup_model(size, weights_dir), generate(...) — plus a batched generate_batch(...).

OPTIMIZATIONS (all measured to matter on CPU):
  1. jit          — generate() compiles model.apply once (public region; invisible to restrict).
  2. static cache — fixed-size KV cache written in place, so the compiled program is reused every
                    decode step instead of recompiling as the cache grows.
  3. batching     — the einsums already carry a batch axis; generate_batch runs many prompts at once,
                    which is the big CPU throughput win (amortizes the weight reads).

RESTRICT NOTE: the private architecture still uses ONLY allow-listed constructs and the SAME policy
as before (same allow_functions -> same policy_id). The one new mechanical op — writing the KV
cache in place with jax.lax.dynamic_update_slice — lives in the PUBLIC wrapper `cache_write`, which
the private code calls by name (like the existing shape_of / append_to / _get wrappers). So the
private region gains no new library call, and the certificate's policy_id is unchanged; only its
source hash changes (both owners re-approve the new source once).

The private region carves itself out with `# syft-restrict: ...` markers, so run() takes no ranges.
"""

import os
import time

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import sentencepiece as spm
from flax import linen as nn


# ── Model configs ────────────────────────────────────────────────────────────
# syft-restrict: obfuscate-start
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
# syft-restrict: obfuscate-end


# syft-restrict: obfuscate-start
def _attn_types(num_layers):
    # syft-restrict: hide-start
    pattern = ("local",) * 5 + ("global",)
    return (pattern * ((num_layers + 5) // 6))[:num_layers]
    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# ── Standalone helpers ────────────────────────────────────────────────────


# syft-restrict: obfuscate-start
def apply_rope(x, positions, base_freq):
    # syft-restrict: hide-start
    """Rotary position embeddings (split-half rotation)."""
    half = shape_of(x)[-1] // 2
    freq_exp = (2.0 / shape_of(x)[-1]) * jnp.arange(half, dtype=jnp.float32)
    timescale = base_freq**freq_exp
    angles = positions[..., None, None] / timescale
    sin, cos = jnp.sin(angles), jnp.cos(angles)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# ── Public wrappers (read directly by the data owners) ─────────────────────
# The private region calls these by name; it never performs the wrapped operation itself.


def _get(module, name):
    """Read a pre-loaded param without shape checking."""
    return module.variable("params", name, lambda: None).value


def shape_of(x):
    """Read an array's shape — an attribute read on a value, not allowed in the private region."""
    return x.shape


def append_to(lst, item):
    """Append to a Python list (a named method on a value)."""
    lst.append(item)
    return lst


def cache_write(cache, update, pos):
    """Write `update` into a fixed-size KV cache at sequence position `pos`, in place.

    Static-shape replacement for growing the cache with concatenate: the buffer stays
    [B, max_len, ...] so the compiled decode step is reused every token. Uses
    dynamic_update_slice here (public) so the private region needs no new allow-listed call.

    Casts the update to the buffer dtype: with bf16 weights, k is float32 (RoPE upcasts via its
    float32 sin/cos) while v stays bf16, so a fixed-dtype buffer needs the write coerced.
    """
    return jax.lax.dynamic_update_slice(
        cache, update.astype(cache.dtype), (0, pos, 0, 0)
    )


def attn_masks(write_pos, q_len, max_len, sliding_window, valid_mask):
    """Boolean attention masks over the static cache — mechanical bookkeeping, not architecture.

    Returns {"local", "global"} each shaped [B, 1, q_len, max_len]:
      causal  : key position <= query position          (no attending to the future)
      window  : query - key < sliding_window            (local layers only)
      valid   : key is a real token, not left-padding   (per sequence, from valid_mask)
    """
    key_pos = jnp.arange(max_len)
    q_pos = write_pos + jnp.arange(q_len)
    delta = q_pos[:, None] - key_pos[None, :]  # [q_len, max_len]
    causal = delta >= 0
    window = delta < sliding_window
    vm = valid_mask[:, None, None, :]  # [B, 1, 1, max_len]
    return {
        "local": (causal & window)[None, None] & vm,
        "global": causal[None, None] & vm,
    }


# ── Flax modules ───────────────────────────────────────────────────────────


# syft-restrict: obfuscate-start
class Einsum(nn.Module):
    def setup(self):
        # syft-restrict: hide-start
        self.w = _get(self, "w")

    # syft-restrict: hide-end

    def __call__(self, equation, x):
        # syft-restrict: hide-start
        return jnp.einsum(equation, x, self.w)

    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# syft-restrict: obfuscate-start
class RMSNorm(nn.Module):
    def setup(self):
        # syft-restrict: hide-start
        self.scale = _get(self, "scale")

    # syft-restrict: hide-end

    def __call__(self, x):
        # syft-restrict: hide-start
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(var + 1e-6) * (1 + self.scale)

    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# syft-restrict: obfuscate-start
class Attention(nn.Module):
    cfg: dict

    def setup(self):
        # syft-restrict: hide-start
        self.q_einsum = Einsum()
        self.kv_einsum = Einsum()
        self._query_norm = RMSNorm()
        self._key_norm = RMSNorm()
        self.attn_vec_einsum = Einsum()

    # syft-restrict: hide-end

    def __call__(self, x, positions, mask, attn_type, cache_k, cache_v, write_pos):
        # syft-restrict: hide-start
        q = self.q_einsum("bsd,ndh->bsnh", x)
        kv = self.kv_einsum("bsd,ckdh->cbskh", x)
        k, v = kv[0], kv[1]

        q = self._query_norm(q)
        k = self._key_norm(k)

        base = LOCAL_ROPE_BASE if attn_type == "local" else GLOBAL_ROPE_BASE
        q = apply_rope(q, positions, base)
        k = apply_rope(k, positions, base)

        # write new keys/values into the fixed-size cache at the current position (public wrapper)
        cache_k = cache_write(cache_k, k, write_pos)
        cache_v = cache_write(cache_v, v, write_pos)

        q = q * (self.cfg["head_dim"] ** -0.5)

        repeats = self.cfg["num_heads"] // self.cfg["num_kv_heads"]
        k_exp = jnp.repeat(cache_k, repeats, axis=2)
        v_exp = jnp.repeat(cache_v, repeats, axis=2)

        logits = jnp.einsum("bsnh,btnh->bnst", q, k_exp)
        logits = jnp.where(mask, logits, K_MASK)
        weights = jax.nn.softmax(logits, axis=-1)

        out = jnp.einsum("bnst,btnh->bsnh", weights, v_exp)
        return self.attn_vec_einsum("bsnh,nhd->bsd", out), cache_k, cache_v

    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# syft-restrict: obfuscate-start
class FeedForward(nn.Module):
    def setup(self):
        # syft-restrict: hide-start
        self.gating_einsum = Einsum()
        self.linear = Einsum()

    # syft-restrict: hide-end

    def __call__(self, x):
        # syft-restrict: hide-start
        gate = self.gating_einsum("bsf,nhf->bsnh", x)
        h = jax.nn.gelu(gate[:, :, 0, :]) * gate[:, :, 1, :]
        return self.linear("bsh,hf->bsf", h)

    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# syft-restrict: obfuscate-start
class Block(nn.Module):
    cfg: dict
    attn_type: str = "local"

    def setup(self):
        # syft-restrict: hide-start
        self.pre_attention_norm = RMSNorm()
        self.attn = Attention(cfg=self.cfg)
        self.post_attention_norm = RMSNorm()
        self.pre_ffw_norm = RMSNorm()
        self.mlp = FeedForward()
        self.post_ffw_norm = RMSNorm()

    # syft-restrict: hide-end

    def __call__(self, x, positions, mask, cache_k, cache_v, write_pos):
        # syft-restrict: hide-start
        h = self.pre_attention_norm(x)
        h, cache_k, cache_v = self.attn(
            h, positions, mask, self.attn_type, cache_k, cache_v, write_pos
        )
        h = self.post_attention_norm(h)
        x = x + h
        h = self.pre_ffw_norm(x)
        h = self.mlp(h)
        h = self.post_ffw_norm(h)
        return x + h, cache_k, cache_v

    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# syft-restrict: obfuscate-start
class Embedder(nn.Module):
    cfg: dict

    def setup(self):
        # syft-restrict: hide-start
        self.input_embedding = _get(self, "input_embedding")

    # syft-restrict: hide-end

    def __call__(self, token_ids):
        # syft-restrict: hide-start
        table = self.input_embedding
        return table[token_ids] * jnp.sqrt(float(self.cfg["embed_dim"])), table

    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


# syft-restrict: obfuscate-start
class Transformer(nn.Module):
    cfg: dict

    def setup(self):
        # syft-restrict: hide-start
        num_layers = self.cfg["num_layers"]
        attn_types = _attn_types(num_layers)
        self.embedder = Embedder(cfg=self.cfg)
        self.layer = [
            Block(cfg=self.cfg, attn_type=attn_types[i]) for i in range(num_layers)
        ]
        self.final_norm = RMSNorm()

    # syft-restrict: hide-end

    def __call__(self, tokens, cache_k, cache_v, write_pos, valid_mask):
        # syft-restrict: hide-start
        sliding_window = self.cfg["sliding_window"]
        num_layers = self.cfg["num_layers"]
        attn_types = _attn_types(num_layers)

        q_len = shape_of(tokens)[1]
        max_len = shape_of(valid_mask)[1]
        positions = write_pos + jnp.arange(q_len)
        masks = attn_masks(write_pos, q_len, max_len, sliding_window, valid_mask)

        x, embed_table = self.embedder(tokens)

        new_k, new_v = [], []
        for i in range(num_layers):
            block = self.layer[i]
            x, kc, vc = block(
                x, positions, masks[attn_types[i]], cache_k[i], cache_v[i], write_pos
            )
            new_k = append_to(new_k, kc)
            new_v = append_to(new_v, vc)

        x = self.final_norm(x[:, -1:])  # last position only
        logits = x @ jnp.transpose(embed_table)  # [B, 1, VOCAB]
        return logits, new_k, new_v

    # syft-restrict: hide-end


# syft-restrict: obfuscate-end


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


def setup_model(size, weights_dir):
    """Configure model, load weights and tokenizer. Returns (model, tokenizer, params)."""
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


def empty_cache(cfg, batch, max_len):
    """Allocate a fixed-size KV cache: one [B, max_len, KVH, hd] buffer per layer, for k and v."""
    kvh, hd, layers = cfg["num_kv_heads"], cfg["head_dim"], cfg["num_layers"]

    def zeros():
        return jnp.zeros((batch, max_len, kvh, hd), jnp.float32)

    return [zeros() for _ in range(layers)], [zeros() for _ in range(layers)]


_APPLY_CACHE = {}


def _jitted_apply(model):
    """jit model.apply once per model and reuse it, so repeated generate_batch() calls (chunks)
    don't recompile — model.apply is a fresh bound method each access, so cache the wrapper."""
    key = id(model)
    if key not in _APPLY_CACHE:
        _APPLY_CACHE[key] = jax.jit(model.apply)
    return _APPLY_CACHE[key]


def generate_batch(model, params, sp, prompts, max_new_tokens=100, max_len=None):
    """Greedy batched generation with a static KV cache and one jit-compiled step.

    prompts: list[str]. Returns (list[str] completions, stats dict).
    Left-pads prompts to a common length so every sequence's real tokens are right-aligned and
    decoding continues from the same position; left-padding is masked out in attention.
    """
    eos, bos = sp.eos_id(), sp.bos_id()
    seqs = [[bos] + sp.EncodeAsIds(format_chat(p)) for p in prompts]
    lens = [len(s) for s in seqs]
    prompt_len = max(lens)
    if max_len is None:
        max_len = prompt_len + max_new_tokens

    # left-pad to prompt_len; valid_mask is True from each sequence's first real token onward
    tokens = jnp.asarray(
        [[0] * (prompt_len - n) + s for s, n in zip(seqs, lens)], jnp.int32
    )
    valid_mask = jnp.asarray(
        [[j >= (prompt_len - n) for j in range(max_len)] for n in lens]
    )

    step = _jitted_apply(
        model
    )  # compiled once per model, reused across chunks and decode steps
    ck, cv = empty_cache(model.cfg, len(prompts), max_len)

    t0 = time.time()
    logits, ck, cv = step(params, tokens, ck, cv, jnp.asarray(0, jnp.int32), valid_mask)
    nxt = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)  # [B]
    jax.block_until_ready(nxt)
    ttft = time.time() - t0

    collected = [nxt]
    t1 = time.time()
    for i in range(max_new_tokens - 1):
        logits, ck, cv = step(
            params,
            nxt[:, None],
            ck,
            cv,
            jnp.asarray(prompt_len + i, jnp.int32),
            valid_mask,
        )
        nxt = jnp.argmax(logits[:, -1], axis=-1).astype(jnp.int32)
        collected.append(nxt)
    gen = jnp.stack(collected, axis=1)  # [B, max_new_tokens]
    jax.block_until_ready(gen)
    decode_elapsed = time.time() - t1
    gen = gen.tolist()

    results = []
    for row in gen:
        ids = row[: row.index(eos)] if eos in row else row
        text = sp.Decode(ids).split("<end_of_turn>")[0].strip()
        results.append(text)

    n_decode = max_new_tokens - 1
    stats = {
        "ttft": ttft,
        "decode_tps": (len(prompts) * n_decode) / decode_elapsed
        if decode_elapsed > 0
        else 0.0,
        "batch": len(prompts),
        "max_new_tokens": max_new_tokens,
    }
    return results, stats


def generate(model, params, sp, prompt, max_new_tokens=100, **kwargs):
    """Single-prompt convenience wrapper around generate_batch (keeps the old call site working).

    Note: this optimized engine uses greedy decoding (deterministic). temperature/top_k sampling
    can be added in generate_batch; it does not affect the private region or restrict.
    """
    results, stats = generate_batch(
        model, params, sp, [prompt], max_new_tokens=max_new_tokens
    )
    return results[0], stats
