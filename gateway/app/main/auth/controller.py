from .worker import Worker


class WorkerController:
    """ This class implements controller design pattern over the workers."""

    def __init__(self):
        self.workers = {}

    def create_worker(self, worker_id: str):
        """ Register a new worker
            Args:
                worker_id: id used to identify the new worker.
            Returns:
                worker: a Worker instance.
        """
        worker = Worker(worker_id)
        self.workers[worker.worker_id] = worker
        return self.workers[worker.worker_id]

    def delete_worker(self, worker_id):
        """ Remove a registered worker.
            Args:
                worker_id: Id used identify the desired worker. 
        """
        del self.workers[worker_id]

    def get_worker(self, worker_id):
        """ Retrieve the desired worker.
            Args:
                worker_id: Id used to identify the desired worker.
            Returns:
                worker: worker Instance or None if it wasn't found.
        """
        return self.workers.get(worker_id, None)
