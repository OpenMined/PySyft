import syft
from typing import Final
from .policy import Policy
from . import register_policy

MAX_DEFAULT_MEMORY: Final = 25


class MemoryLimitPolicy(Policy):
    @syft.typecheck.type_hints
    def __init__(self, max_objs: int):
        self.max_objs = max_objs

    @syft.typecheck.type_hints
    def error_policy(self, stats: "syft.worker.supervisor.stats.WorkerStats") -> None:
        raise MemoryError("Memory limit exceeded.")

    @syft.typecheck.type_hints
    def enforce_policy(self, stats: "syft.worker.supervisor.stats.WorkerStats") -> None:
        pass


register_policy(MemoryLimitPolicy(MAX_DEFAULT_MEMORY))
