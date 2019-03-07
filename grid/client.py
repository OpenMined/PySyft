from syft.workers import WebsocketClientWorker
import requests


class GridClient(WebsocketClientWorker):
    def __init__(
        self, hook, host, port, grid, pid, id=0, log_msgs=False, verbose=False, data={}
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """
        self.flask_port = 5000
        self.grid = grid
        self.pid = pid

        super().__init__(
            hook=hook,
            host=host,
            port=port,
            id=id,
            data=data,
            log_msgs=log_msgs,
            verbose=verbose,
        )

    @property
    def known_workers(self):
        url = "http://" + self.host + ":" + str(self.flask_port) + "/get_known_workers/"

        response = requests.get(url)

        try:
            response = response.json()
        except:
            return response.text
        return response
