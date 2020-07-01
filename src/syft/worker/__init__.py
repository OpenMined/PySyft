from .virtual.virtual_worker import VirtualWorker
from .worker import Worker


def create_virtual_workers(*args):
    clients = list()
    for worker_name in args:
        clients.append(VirtualWorker(worker_name).get_client())
    return clients
