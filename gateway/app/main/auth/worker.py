class Worker:
    """ An abstraction of a worker. """

    def __init__(self, worker_id: str):
        """ Create a worker instance.
            
            Args:
                worker_id: the id that uniquely identifies the user.
        """
        self._worker_id = worker_id

    def worker_id(self) -> str:
        """ Get the id of this worker.
            Returns:
                worker_id (str) : the id of this worker.
        """
        return self._worker_id
