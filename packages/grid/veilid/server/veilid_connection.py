# stdlib
from collections.abc import Callable

# third party
import veilid
from veilid import VeilidUpdate
from veilid.json_api import _JsonRoutingContext
from veilid.json_api import _JsonVeilidAPI

# relative
from .constants import HOST
from .constants import PORT
from .constants import USE_DIRECT_CONNECTION


async def noop_callback(update: VeilidUpdate) -> None:
    pass


async def get_veilid_conn(
    host: str = HOST, port: int = PORT, update_callback: Callable = noop_callback
) -> _JsonVeilidAPI:
    return await veilid.json_api_connect(
        host=host, port=port, update_callback=update_callback
    )


async def get_routing_context(conn: _JsonVeilidAPI) -> _JsonRoutingContext:
    if USE_DIRECT_CONNECTION:
        return await (await conn.new_routing_context()).with_safety(
            veilid.SafetySelection.unsafe(veilid.Sequencing.ENSURE_ORDERED)
        )
    else:
        return await (await conn.new_routing_context()).with_sequencing(
            veilid.Sequencing.ENSURE_ORDERED
        )
