# grid relative
from ...exceptions import ProtocolNotFoundError
from ...manager.database_manager import DatabaseManager
from .protocol import Protocol


class ProtocolManager(DatabaseManager):
    schema = Protocol

    def __init__(self, database):
        self.db = database
        self._schema = ProtocolManager.schema
        # self._protocols = DatabaseManager(Protocol)

    def register(self, process, protocols: dict):
        # Register new Protocols into the database
        for key, value in protocols.items():
            super().register(name=key, value=value, protocol_flprocess=process)

    def get(self, **kwargs):
        """Retrieve the desired protocol.

        Args:
            query : query used to identify the desired protcol object.
        Returns:
            plan : Protocol Instance or None if it wasn't found.
        Raises:
            ProtocolNotFound (PyGridError) : If Protocol not found.
        """
        _protocol = self.query(**kwargs)

        if not _protocol:
            raise ProtocolNotFoundError
        return _protocol

    # def delete(self, **kwargs):
    #     """Delete a registered Protocol.

    #     Args:
    #         query: Query used to identify the protocol object.
    #     """
    #     self._protocols.delete(**kwargs)
