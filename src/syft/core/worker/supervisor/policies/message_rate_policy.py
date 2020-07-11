
from .policy import Policy
from . import register_policy
from .....typecheck import type_hints
from ..stats import WorkerStats

MAX_RATE_DEFAULT = 100


class MessageRatePolicy(Policy):
    @type_hints
    def __init__(self, max_frequency: float):
        self.max_frequency = max_frequency

    @type_hints
    def error_policy(self, stats: WorkerStats) -> None:
        raise MemoryError("Memory limit exceeded.")

    @type_hints
    def enforce_policy(self, stats: WorkerStats) -> None:
        pass


register_policy(policy=MessageRatePolicy(max_frequency=MAX_RATE_DEFAULT))
