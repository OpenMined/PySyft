"""Inference service: loads the model from synced weights and serves requests.

The backend is injected (anything exposing ``load(model_size, weights_dir)``
and ``generate(loaded, prompt, max_new_tokens)``) so tests can use a stub
instead of the real ``gemma`` backend.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Callable

from enclave_model_api.log_writer import append_log_record, build_log_record
from enclave_model_api.paths import weights_ready

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(
        self,
        backend,
        model_size: str,
        weights_dir: Path | str | Callable[[], Path],
        logs_dir: Path | str,
    ):
        self.backend = backend
        self.model_size = model_size
        # A callable is re-resolved on each read. The weights arrive in the
        # layout their owner writes, so a path fixed at startup can watch a
        # layout the owner never writes, and the poll then never ends.
        self._weights_dir = (
            weights_dir if callable(weights_dir) else (lambda: Path(weights_dir))
        )
        self.logs_dir = Path(logs_dir)
        self._loaded = None
        self._lock = threading.Lock()

    @property
    def weights_dir(self) -> Path:
        return Path(self._weights_dir())

    @property
    def loaded(self) -> bool:
        return self._loaded is not None

    @property
    def weights_present(self) -> bool:
        return weights_ready(self.weights_dir)

    def try_load(self) -> bool:
        """Load the model once it's ready. Returns loaded state.

        Backends that need real weights (``requires_weights``) only load once the
        weights have synced; the mock backend loads immediately.
        """
        if self.loaded:
            return True
        if getattr(self.backend, "requires_weights", True) and not self.weights_present:
            return False
        logger.info("Loading %s model from %s", self.model_size, self.weights_dir)
        self._loaded = self.backend.load(self.model_size, self.weights_dir)
        logger.info("Model loaded")
        return True

    def start_polling(self, interval: float = 5.0) -> threading.Thread:
        """Poll for weights in a daemon thread until the model is loaded."""
        thread = threading.Thread(
            target=self._poll_until_loaded, args=(interval,), daemon=True
        )
        thread.start()
        return thread

    def _poll_until_loaded(self, interval: float) -> None:
        while not self.loaded:
            try:
                if self.try_load():
                    return
            except Exception:
                logger.exception("Model load failed — retrying")
            time.sleep(interval)

    def infer(self, prompt: str, max_new_tokens: int = 200) -> tuple[str, dict]:
        """Run one inference and append its log record. Serialized by a lock."""
        with self._lock:
            completion, stats = self.backend.generate(
                self._loaded, prompt, max_new_tokens=max_new_tokens
            )
            append_log_record(
                self.logs_dir, build_log_record(prompt, completion, stats)
            )
        return completion, stats
