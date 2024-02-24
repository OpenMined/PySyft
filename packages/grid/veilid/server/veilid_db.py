# Contains all the database related functions for the Veilid server
# stdlib
from typing import Optional

# third party
from veilid import KeyPair
from veilid import TypedKey
from veilid.json_api import _JsonVeilidAPI

# relative
from .constants import DHT_KEY
from .constants import DHT_KEY_CREDS
from .constants import TABLE_DB_KEY


async def load_key(conn: _JsonVeilidAPI, key: str) -> Optional[str]:
    tdb = await conn.open_table_db(TABLE_DB_KEY, 1)

    async with tdb:
        key_bytes = key.encode()
        value = await tdb.load(key_bytes)
        if value is None:
            return None
        return value.decode()


async def store_key(conn: _JsonVeilidAPI, key: str, value: str) -> None:
    tdb = await conn.open_table_db(TABLE_DB_KEY, 1)

    async with tdb:
        key_bytes = key.encode()
        value_bytes = value.encode()
        await tdb.store(key_bytes, value_bytes)


async def load_dht_key(conn: _JsonVeilidAPI) -> Optional[TypedKey]:
    value = await load_key(conn, DHT_KEY)
    if value is None:
        return None
    return TypedKey(value)


async def load_dht_key_creds(conn: _JsonVeilidAPI) -> Optional[KeyPair]:
    value = await load_key(conn, DHT_KEY_CREDS)
    if value is None:
        return None
    return KeyPair(value)


async def store_dht_key(conn: _JsonVeilidAPI, keypair: TypedKey) -> None:
    await store_key(conn, DHT_KEY, str(keypair))


async def store_dht_key_creds(conn: _JsonVeilidAPI, keypair: KeyPair) -> None:
    await store_key(conn, DHT_KEY_CREDS, str(keypair))
