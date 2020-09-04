from ..codes import WORKER_PROPERTIES
from ..workers import Worker


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class SocketHandler(metaclass=Singleton):
    """Socket Handler is a singleton class used to handle/manage websocket
    connections."""

    def __init__(self):
        self.connections = {}

    def new_connection(self, workerId: str, socket):
        """Create a mapping structure to establish a bond between a workerId
        and a socket descriptor.

        Args:
            workerId: UUID string used to identify workers.
            socket: Socket descriptor that will be used to send/receive messages from this client.

        Returns:
            Worker: a worker instance with the corresponding workerId
        """
        if workerId not in self.connections:
            self.connections[workerId] = Worker(workerId, socket)
        else:
            worker = self.connections[workerId]
            if worker.status == WORKER_PROPERTIES.OFFLINE:
                worker._socket = socket
        return self.connections[workerId]

    def send_msg(self, workerId: str, message: str):
        """Find the socket descriptor mapped by workerId and send them a
        message.

        Args:
            workerId: UUID string used to identify and retrieve a worker.
            message: Message to be send.
        """
        socket = self.connections.get(workerId, None)
        if socket:
            socket.send(message)

    def get(self, query):
        """Retrieve a worker by its UUID string or its socket descriptor."""
        if isinstance(query, str):
            return self.connections.get(query, None)
        else:
            return self.__retrieve_worker_by_socket(query)

    def remove(self, socket) -> str:
        """Remove a socket descriptor from mapping structure. It will be used
        when the socket connection is closed.

        Args:
            socket: socket descriptor used to send/receive messages.

        Returns:
            workerId: Worker id linked to that connection.
        """
        worker = self.__retrieve_worker_by_socket(socket)
        if worker:
            self.connections[worker._id]._socket = None
            self.connections[worker._id].connected_nodes = []
            return worker._id

    def __retrieve_worker_by_socket(self, socket):
        for worker_id, worker in self.connections.items():
            if worker._socket == socket:
                return self.connections[worker_id]

    @property
    def nodes(self) -> list:
        """Return all the connected nodes as a list of tuples of (worker_id,
        worker)"""
        return list(self.connections.items())

    def __len__(self) -> int:
        """Number of connections handled by this server.

        Returns:
            length: number of connections handled by this server.
        """
        return len(self.connections)
