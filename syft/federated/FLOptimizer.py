import copy

#To create multiple instances of optimizers for FL

class Optims:
    def __init__(self, workers, optim):
        self.optim = optim
        self.workers = workers
        self.optimizers = {}
        for worker in workers:
            '''
            self.optimizers[str(worker)] = self.optim will not work as it points to the same instance of the object.
            copy.deepcopy not works as the parameters wont get updated.
            '''
            self.optimizers[str(worker)] = copy.copy(self.optim)
            self.optimizers[str(worker)].load_state_dict((self.optim).state_dict())

    def get_optim(self, worker):
        return self.optimizers[str(worker)]   