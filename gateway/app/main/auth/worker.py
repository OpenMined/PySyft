class Worker:
    """ An abstraction of a worker. """

    def __init__(self, worker_id: str):
        """ Create a worker instance.
            Args:
                worker_id: the id that uniquely identifies the user.
        """
        self._worker_id = worker_id
        self._cycles = {}

    def register_cycle(self, hash_key: str, fl_cycle, model_id) -> bool:
        """ Save a new Cycle in worker's cycle registry by hash code.
            Args:
                hash_key: Key used to map the cycle.
                fl_cycle: FederatedLearning Cycle instance.
        """
        self._cycle[hash_key] = fl_cycle

    def get_cycle(self, hash_key: str):
        """ Retrieve a specific cycle mapped by hash key.
            Args:
                hash_key : Hash key used to identify the desired cycle.
            Retrurns:
                cycle: A Cycle instance of None (if not found).
        """
        return self._cycle.get(hash_key, None)

    def worker_id(self) -> str:
        """ Get the id of this worker.
            Returns:
                worker_id (str) : the id of this worker.
        """
        return self._worker_id
