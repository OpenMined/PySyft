from .worker import Worker
from ..storage.warehouse import Warehouse
from ..storage import models
from ..exceptions import WorkerNotFoundError


class WorkerController:
    """ This class implements controller design pattern over the workers."""

    def __init__(self):
        self._workers = Warehouse(models.Worker)

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
        """ Remove a registered worker.
            Args:
                worker_id: Id used identify the desired worker. 
        """
        self._workers.delete(**kwargs)

    def get(self, **kwargs):
        """ Retrieve the desired worker.
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
        """ Update Workers Attributes. """
        return self._workers.update()
