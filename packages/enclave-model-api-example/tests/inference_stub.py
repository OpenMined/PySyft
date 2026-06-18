"""Stub inference backend + weight fixtures shared by the inference tests.

Implements the same interface as enclave_model_api.backend.GemmaBackend
(load/generate) without any model deps.
"""

from pathlib import Path

STUB_COMPLETION_PREFIX = "[stub response to:"


class StubBackend:
    def load(self, model_size: str, weights_dir: Path | str):
        weights_dir = Path(weights_dir)
        assert (weights_dir / "tokenizer.model").is_file()
        return {"model_size": model_size}

    def generate(
        self, loaded, prompt: str, max_new_tokens: int = 200
    ) -> tuple[str, dict]:
        completion = f"{STUB_COMPLETION_PREFIX} {prompt[:40]}]"
        return completion, {"elapsed": 0.01, "max_new_tokens": max_new_tokens}


def make_stub_weights(weights_dir: Path | str) -> Path:
    """Create the expected weights layout: tokenizer.model + one checkpoint dir."""
    weights_dir = Path(weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    (weights_dir / "tokenizer.model").write_bytes(b"stub_tokenizer")
    ckpt_dir = weights_dir / "gemma-3-270m-it"
    ckpt_dir.mkdir(exist_ok=True)
    (ckpt_dir / "checkpoint").write_bytes(b"stub_checkpoint_data")
    return weights_dir
