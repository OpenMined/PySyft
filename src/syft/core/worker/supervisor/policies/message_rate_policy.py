import syft
from .policy import Policy
from . import register_policy

MAX_RATE_DEFAULT = 100


class MessageRatePolicy(Policy):
    @syft.typecheck.type_hints
    def __init__(self, max_frequency: float):
        self.max_frequency = max_frequency

    @syft.typecheck.type_hints
    def error_policy(self, stats: "syft.worker.supervisor.stats.WorkerStats") -> None:
        raise MemoryError("Memory limit exceeded.")

    @syft.typecheck.type_hints
    def enforce_policy(self, stats: "syft.worker.supervisor.stats.WorkerStats") -> None:
        pass


register_policy(MessageRatePolicy(MAX_RATE_DEFAULT))
