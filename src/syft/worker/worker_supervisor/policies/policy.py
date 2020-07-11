from __future__ import annotations
from ..stats import WorkerStats
from ..... import type_hints


class Policy:
    @type_hints
    def error_policy(stats: WorkerStats) -> None:
        raise NotImplementedError

    @type_hints
    def enforce_policy(stats: WorkerStats) -> None:
        raise NotImplementedError
