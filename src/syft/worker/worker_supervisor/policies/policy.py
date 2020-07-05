from __future__ import annotations
import syft

class Policy:
    @syft.typecheck.type_hints
    def error_policy(stats: "syft.worker.worker_supervisor.stats.WorkerStats") -> None:
        raise NotImplementedError

    @syft.typecheck.type_hints
    def enforce_policy(stats: "syft.worker.worker_supervisor.stats.WorkerStats") -> None:
        raise NotImplementedError
