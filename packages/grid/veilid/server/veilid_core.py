# stdlib
from typing import Callable
from typing import Optional

# third party
import veilid
from veilid import VeilidUpdate
from veilid.json_api import _JsonVeilidAPI

# relative
from .constants import HOST
from .constants import PORT


async def main_callback(update: VeilidUpdate) -> None:
    print(update)


async def noop_callback(update: VeilidUpdate) -> None:
    pass


async def get_veilid_conn(
    host: str = HOST, port: int = PORT, update_callback: Callable = noop_callback
) -> _JsonVeilidAPI:
    return await veilid.json_api_connect(
        host=HOST, port=PORT, update_callback=noop_callback
    )


class VeilidConnectionSingleton:
    _instance = None

    def __new__(cls) -> "VeilidConnectionSingleton":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connection = None
        return cls._instance

    def __init__(self) -> None:
        self._connection: Optional[_JsonVeilidAPI] = None

    @property
    def connection(self) -> Optional[_JsonVeilidAPI]:
        return self._connection

    async def initialize_connection(self) -> None:
        if self._connection is None:
            self._connection = await get_veilid_conn(update_callback=main_callback)
            # TODO: Shift to Logging Module
            print("Connected to Veilid")

    async def release_connection(self) -> None:
        if self._connection is not None:
            await self._connection.release()
            # TODO: Shift to Logging Module
            print("Disconnected from Veilid")
            self._connection = None
