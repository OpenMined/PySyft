class BaseService(object):
    def __init__(self, worker):
        self.worker = worker
        self.api = self.worker.api
