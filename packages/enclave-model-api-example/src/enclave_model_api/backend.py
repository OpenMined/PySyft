"""Gemma backend — the only module importing model dependencies.

Uses the official DeepMind ``gemma`` package (JAX) so the whole model is
loaded through its high-level API; no architecture code lives here. Only
importable where the docker-only model deps are installed (see
docker/requirements.txt).

Weights-dataset layout (uploaded by the model owner):
    <dataset>/tokenizer.model    SentencePiece tokenizer
    <dataset>/<checkpoint>/      Orbax checkpoint directory (exactly one subdir)
"""

import time
from pathlib import Path

from gemma import gm

from enclave_model_api.paths import find_checkpoint_dir

MODEL_CLASSES = {
    "270m": gm.nn.Gemma3_270M,
    "1b": gm.nn.Gemma3_1B,
    "4b": gm.nn.Gemma3_4B,
}


class GemmaBackend:
    """Implements the inference-backend interface: load() and generate()."""

    def load(self, model_size: str, weights_dir: Path | str):
        weights_dir = Path(weights_dir)
        model = MODEL_CLASSES[model_size]()
        params = gm.ckpts.load_params(find_checkpoint_dir(weights_dir))
        tokenizer = gm.text.Gemma3Tokenizer(path=str(weights_dir / "tokenizer.model"))
        return gm.text.ChatSampler(model=model, params=params, tokenizer=tokenizer)

    def generate(
        self, sampler, prompt: str, max_new_tokens: int = 200
    ) -> tuple[str, dict]:
        started = time.time()
        completion = sampler.chat(prompt, max_new_tokens=max_new_tokens)
        elapsed = time.time() - started
        return completion, {"elapsed": round(elapsed, 3)}
