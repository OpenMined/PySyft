# Syft assets module imports
from ...core.exceptions import ProtocolNotFoundError

# PyGrid imports
from ...core.warehouse import Warehouse
from .protocol import Protocol


class ProtocolManager:
    def __init__(self):
        self._protocols = Warehouse(Protocol)

    def register(self, process, protocols: dict):
        # Register new Protocols into the database
        for key, value in protocols.items():
            self._protocols.register(name=key, value=value, protocol_flprocess=process)

    def get(self, **kwargs):
        """Retrieve the desired protocol.

        Args:
            query : query used to identify the desired protcol object.
        Returns:
            plan : Protocol Instance or None if it wasn't found.
        Raises:
            ProtocolNotFound (PyGridError) : If Protocol not found.
        """
        _protocol = self._protocols.query(**kwargs)

        if not _protocol:
            raise ProtocolNotFoundError
        return _protocol

    def delete(self, **kwargs):
        """Delete a registered Protocol.

        Args:
            query: Query used to identify the protocol object.
        """
        self._protocols.delete(**kwargs)
