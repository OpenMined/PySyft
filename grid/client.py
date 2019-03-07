from syft.workers import WebsocketClientWorker
import requests


class GridClient(WebsocketClientWorker):
    def __init__(
        self,
        hook,
        host,
        port,
        grid,
        pid,
        id=0,
        log_msgs=False,
        verbose=False,
        data={},
        flask_port=5000,
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """
        self.flask_port = flask_port
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

    def add_known_worker(self, host, port):

        url = (
            "http://"
            + self.host
            + ":"
            + str(self.flask_port)
            + "/add_worker/"
            + str(host)
            + "/"
            + str(port)
        )

        response = requests.get(url)

        try:
            response = response.json()
            if "error" in response:
                print(response)
                return self.known_workers
            return response

        except:
            return response.text

        return response

    def __del__(self):
        self.grid.kill_worker(self)
