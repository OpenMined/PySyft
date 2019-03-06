from syft.workers import WebsocketClientWorker

class GridClient(WebsocketClientWorker):
    def __init__(
            self, hook, host, port, grid, pid, id=0, log_msgs=False, verbose=False, data={}
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """

        self.grid = grid
        self.pid = pid

        super().__init__(hook=hook,
                         host=host,
                         port=port,
                         id=id,
                         data=data,
                         log_msgs=log_msgs,
                         verbose=verbose)
