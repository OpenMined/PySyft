# third party
from loguru import logger
from veilid.json_api import _JsonVeilidAPI

# relative
from .veilid_callback import main_callback
from .veilid_connection import get_veilid_conn


class VeilidConnectionSingleton:
    _instance = None

    def __new__(cls) -> "VeilidConnectionSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connection = None
        return cls._instance

    def __init__(self) -> None:
        self._connection: _JsonVeilidAPI | None = None

    @property
    def connection(self) -> _JsonVeilidAPI | None:
        return self._connection

    async def initialize_connection(self) -> None:
        if self._connection is None:
            self._connection = await get_veilid_conn(update_callback=main_callback)
            logger.info("Connected to Veilid")

    async def release_connection(self) -> None:
        if self._connection is not None:
            await self._connection.release()
            logger.info("Disconnected  from Veilid")
            self._connection = None
