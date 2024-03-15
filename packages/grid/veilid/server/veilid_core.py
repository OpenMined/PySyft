# third party
from loguru import logger
import veilid
from veilid import KeyPair
from veilid import Sequencing
from veilid import Stability
from veilid import TypedKey
from veilid import ValueData
from veilid.json_api import _JsonRoutingContext
from veilid.json_api import _JsonVeilidAPI
from veilid.types import RouteId

# relative
from .constants import USE_DIRECT_CONNECTION
from .veilid_connection import get_routing_context
from .veilid_connection import get_veilid_conn
from .veilid_db import load_dht_key
from .veilid_db import store_dht_key
from .veilid_db import store_dht_key_creds


async def create_private_route(
    conn: _JsonVeilidAPI,
    stability: Stability = veilid.Stability.RELIABLE,
    sequencing: Sequencing = veilid.Sequencing.ENSURE_ORDERED,
) -> tuple[RouteId, bytes]:
    route_id, route_blob = await conn.new_custom_private_route(
        [veilid.CryptoKind.CRYPTO_KIND_VLD0],
        stability=stability,
        sequencing=sequencing,
    )
    logger.info(f"Private Route created with Route ID: {route_id}")
    return (route_id, route_blob)


async def get_node_id() -> str:
    logger.info("Getting Node ID")
    # TODO: Cache NODE ID Retrieval
    async with await get_veilid_conn() as conn:
        state = await conn.get_state()
        config = state.config.config
        node_id = config.network.routing_table.node_id[0]
        if not node_id:
            raise Exception("Node ID not found.Veilid might not be ready")
        return node_id


async def generate_dht_key() -> str:
    logger.info("Generating DHT Key")

    async with await get_veilid_conn() as conn:
        if await load_dht_key(conn):
            return "DHT Key already exists"

        async with await get_routing_context(conn) as router:
            dht_record = await router.create_dht_record(veilid.DHTSchema.dflt(1))

            _, route_blob = await create_private_route(conn)
            await router.set_dht_value(dht_record.key, 0, route_blob)

            await router.close_dht_record(dht_record.key)

            keypair = KeyPair.from_parts(
                key=dht_record.owner, secret=dht_record.owner_secret
            )

            await store_dht_key(conn, dht_record.key)
            await store_dht_key_creds(conn, keypair)

    return "DHT Key generated successfully"


async def retrieve_dht_key() -> str:
    async with await get_veilid_conn() as conn:
        dht_key = await load_dht_key(conn)

        if dht_key is None:
            raise Exception("DHT Key does not exist. Please generate one.")
        return str(dht_key)


async def generate_vld_key() -> str:
    if USE_DIRECT_CONNECTION:
        await get_node_id()
    else:
        await generate_dht_key()

    return "Veilid Key generated successfully"


async def retrieve_vld_key() -> str:
    if USE_DIRECT_CONNECTION:
        return await get_node_id()
    else:
        return await retrieve_dht_key()


async def get_dht_value(
    router: _JsonRoutingContext,
    dht_key: TypedKey,
    subkey: int,
    force_refresh: bool = True,
) -> ValueData:
    try:
        await router.open_dht_record(key=dht_key, writer=None)
    except Exception as e:
        raise Exception(f"Unable to open DHT Record:{dht_key} . Exception: {e}")

    try:
        dht_value = await router.get_dht_value(
            key=dht_key, subkey=subkey, force_refresh=force_refresh
        )
        # NOTE: Always close the DHT record after reading the value
        await router.close_dht_record(dht_key)
        return dht_value
    except Exception as e:
        raise Exception(
            f"Unable to get subkey value:{subkey} from DHT Record:{dht_key}. Exception: {e}"
        )


# TODO: change verbosity of logs to debug at appropriate places
async def get_route_from_vld_key(
    vld_key: str, conn: _JsonVeilidAPI, router: _JsonRoutingContext
) -> str | RouteId:
    if USE_DIRECT_CONNECTION:
        route = vld_key
        logger.info(f"Peer Node ID: {route}")
    else:
        dht_key = veilid.TypedKey(vld_key)
        dht_value = await get_dht_value(router, dht_key, 0)
        logger.info(f"DHT Value:{dht_value}")
        route = await conn.import_remote_private_route(dht_value.data)
        logger.info(f"Private Route of  Peer: {route} ")

    return route


async def app_message(vld_key: str, message: bytes) -> str:
    async with await get_veilid_conn() as conn:
        async with await get_routing_context(conn) as router:
            route = await get_route_from_vld_key(vld_key, conn, router)

            await router.app_message(route, message)

            return "Message sent successfully"


async def app_call(vld_key: str, message: bytes) -> bytes:
    async with await get_veilid_conn() as conn:
        async with await get_routing_context(conn) as router:
            route = await get_route_from_vld_key(vld_key, conn, router)

            result = await router.app_call(route, message)

            return result


# TODO: Modify healthcheck endpoint to check public internet ready
async def healthcheck() -> bool:
    async with await get_veilid_conn() as conn:
        state = await conn.get_state()
        return state.network.started
