class GridError(BaseException):
    def __init__(self, error, status):
        self.status = status
        self.error = error
