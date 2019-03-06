import requests
import time
from .client import GridClient
import syft as sy

class Grid():

    def __init__(self, *workers):
        ""

    def get_worker(self, hostname="localhost", port=5000):
        url = 'http://' + hostname + ':' + str(port) + '/get_connection'
        response = requests.get(url).json()

        time.sleep(2)

        uid = str(response['host']) + ":" + str(response['port'])

        return GridClient(hook=sy.hook,
                          grid=self,
                          id=uid,
                          host=response['host'],
                          port=int(response['port']),
                          pid=response['pid'])

    def kill_worker(self, worker):
        url = 'http://' + worker.host + ':5000/kill_connection/' + str(worker.pid)
        requests.get(url)
