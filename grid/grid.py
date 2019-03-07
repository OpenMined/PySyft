import requests
import time
from .client import GridClient
import syft as sy


class Grid:
    def __init__(self, *workers):
        self.workers = list(workers)

    def get_worker(self, hostname="localhost", port=5000):

        try:
            url = "http://" + hostname + ":" + str(port) + "/get_connection"
            response = requests.get(url).json()
        except ConnectionRefusedError as e:
            raise ConnectionRefusedError(
                "Could not find the worker you asked us to find.",
                " Are you sure the hostname and port are correct?",
            )

        time.sleep(2)

        uid = str(response["host"]) + ":" + str(response["port"])

        worker = GridClient(
            hook=sy.hook,
            grid=self,
            id=uid,
            host=response["host"],
            port=int(response["port"]),
            pid=response["pid"],
        )

        self.workers.append(worker)
        return worker

    def kill_worker(self, worker):
        url = "http://" + worker.host + ":5000/kill_connection/" + str(worker.pid)
        requests.get(url)
