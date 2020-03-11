"""to maintain a list of optimizer objects,
one for each worker and use them in the appropriate context"""
import copy


class Optims:
    """to create a list of optimizer objects"""

    def __init__(self, workers, optim):
        self.optim = optim
        self.workers = workers
        self.optimizers = {}
        for worker in workers:
            self.optimizers[str(worker)] = copy.copy(self.optim)
            self.optimizers[str(worker)].load_state_dict((self.optim).state_dict())

    def get_optim(self, worker):
        """returns optimizer for worker"""
        return self.optimizers[str(worker)]

    def count(self):
        """returns the number of optimizer objects"""
        return len(self.workers)
