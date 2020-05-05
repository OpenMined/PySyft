from syft import dependency_check

from syft.workers.base import BaseWorker

from syft.frameworks.crypten.worker_support import add_support_to_worker, remove_support_from_worker


supported_frameworks = {}

if dependency_check.crypten_available:
    supported_frameworks["crypten"] = {
        "add_support": add_support_to_worker,
        "remove_support": remove_support_from_worker,
    }


def add_support(worker: BaseWorker, framework: str) -> None:
    supported_frameworks[framework]["add_support"](worker)


def remove_support(worker: BaseWorker, framework: str) -> None:
    supported_frameworks[framework]["remove_support"](worker)
