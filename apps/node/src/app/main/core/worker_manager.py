# workers module imports
import abc
import logging

from .exceptions import WorkerNotFoundError

# PyGrid imports
from .warehouse import Warehouse

# from .worker import Worker


class WorkerManager(metaclass=abc.ABCMeta):
    def __init__(self, worker: "Worker") -> None:
        self._workers = Warehouse(worker)

    def create(self, worker_id: str):
        """ Register a new worker
            Args:
                worker_id: id used to identify the new worker.
            Returns:
                worker: a Worker instance.
        """
        new_worker = self._workers.register(id=worker_id)
        return new_worker

    def delete(self, **kwargs):
        """Remove a registered worker.

        Args:
            worker_id: Id used identify the desired worker.
        """
        self._workers.delete(**kwargs)

    def get(self, **kwargs):
        """Retrieve the desired worker.

        Args:
            worker_id: Id used to identify the desired worker.
        Returns:
            worker: worker Instance or None if it wasn't found.
        """
        _worker = self._workers.first(**kwargs)

        if not _worker:
            raise WorkerNotFoundError

        return self._workers.first(**kwargs)

    def update(self, worker):
        """Update Workers Attributes."""
        return self._workers.update()

    def is_eligible(self, worker_id: str, server_config: dict):
        """Check if Worker is eligible to join in an new cycle by using its
        bandwidth statistics.

        Args:
            worker_id : Worker's ID.
            server_config : FL Process Server Config.
        Returns:
            result: Boolean flag.
        """
        _worker = self._workers.first(id=worker_id)
        logging.info(
            f"Checking worker [{_worker}] against server_config [{server_config}]"
        )

        # Check bandwidth
        _comp_bandwidth = (
            "minimum_upload_speed" not in server_config
            or _worker.avg_upload >= server_config["minimum_upload_speed"]
        ) and (
            "minimum_download_speed" not in server_config
            or _worker.avg_download >= server_config["minimum_download_speed"]
        )

        logging.info(f"Result of bandwidth check: {_comp_bandwidth}")
        return _comp_bandwidth
