class WorkerSupervisor:
    def __init__(self, function):
        self.function = function

    def __call__(self, *args, **kwargs):
        self.function(*args, **kwargs)
