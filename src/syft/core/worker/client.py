class Client:
    def __init__(self, id, verbose=False):
        self.id = id
        self.verbose = verbose

    def __repr__(self):
        return f"<Client id:{self.id}>"
