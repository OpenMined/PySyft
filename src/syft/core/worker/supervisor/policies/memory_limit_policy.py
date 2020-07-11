

from typing import Final
from .policy import Policy
from . import register_policy
from .....typecheck import type_hints
from ..stats import WorkerStats

MAX_DEFAULT_MEMORY: Final = 25


class MemoryLimitPolicy(Policy):
    @type_hints
    def __init__(self, max_objs: int):
        self.max_objs = max_objs

    @type_hints
    def error_policy(self, stats: WorkerStats) -> None:
        raise MemoryError("Memory limit exceeded.")

    @type_hints
    def enforce_policy(self, stats: WorkerStats) -> None:
        pass


register_policy(policy=MemoryLimitPolicy(max_objs=MAX_DEFAULT_MEMORY))
