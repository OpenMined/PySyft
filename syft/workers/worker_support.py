from syft import dependency_check

from syft.workers.base import BaseWorker


def add_support(worker: BaseWorker, framework: str) -> None:
    assert framework == "crypten", "Only CrypTen is supported for the moment"
    assert dependency_check.crypten_available, "CrypTen not installed"

    from syft.frameworks.crypten.worker_support import add_support_to_workers

    add_support_to_workers(worker)


def remove_support(worker: BaseWorker, framework: str) -> None:
    assert framework == "crypten", "Only CrypTen is supported for the moment"
    assert dependency_check.crypten_available, "CrypTen not installed"

    from syft.frameworks.crypten.worker_support import remove_support_from_workers

    remove_support_from_workers(worker)
