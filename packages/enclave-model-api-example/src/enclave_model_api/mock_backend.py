"""Mock inference backend — canned, sensible responses, no model deps.

Used by default (``SYFT_ENCLAVE_USE_MOCK_MODEL=true``) so the demo runs without
uploading real weights or importing jax/gemma. Responses are keyword-matched so
they make sense for the example prompts (recipes, safety refusals, factual Q&A).
Implements the same interface as ``backend.GemmaBackend`` — ``load`` / ``generate``
— plus ``requires_weights = False`` so the service loads it without any weights.
"""

import time

REFUSAL = (
    "I can't help with that. Providing instructions for creating weapons designed "
    "to cause mass harm — including bio-weapons — is dangerous and against my "
    "guidelines."
)

BANANA_BREAD = (
    "Banana bread: mash 3 ripe bananas, then mix in 1/3 cup melted butter, 1 egg, "
    "3/4 cup sugar, 1 tsp baking soda and a pinch of salt. Fold in 1.5 cups flour, "
    "pour into a loaf tin and bake at 175°C (350°F) for about 50 minutes."
)


class MockBackend:
    """Returns deterministic, prompt-appropriate completions without a real model."""

    requires_weights = False

    def load(self, model_size, weights_dir):
        # Nothing to load — return a marker so the service treats us as ready.
        return "mock-model"

    def generate(
        self, loaded, prompt: str, max_new_tokens: int = 200
    ) -> tuple[str, dict]:
        started = time.time()
        completion = self._respond(prompt)
        return completion, {"elapsed": round(time.time() - started, 3), "mock": True}

    @staticmethod
    def _respond(prompt: str) -> str:
        p = prompt.lower()
        if "bio-weapon" in p or "bioweapon" in p or "weapon" in p:
            return REFUSAL
        if "bread" in p:
            return BANANA_BREAD
        if "capital" in p and "netherlands" in p:
            return "The capital of the Netherlands is Amsterdam."
        if "capital" in p and "france" in p:
            return "The capital of France is Paris."
        if "capital" in p:
            return "I can answer that for well-known countries — try naming one explicitly."
        return f"[mock] Here is a brief, helpful answer to: {prompt.strip()[:80]}"
