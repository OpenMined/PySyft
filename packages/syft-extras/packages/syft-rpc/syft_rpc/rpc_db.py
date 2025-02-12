from __future__ import annotations

import sqlite3
import threading
from functools import cache
from uuid import UUID

from syft_core.client_shim import Client
from typing_extensions import Optional, Union

from syft_rpc.protocol import SyftBulkFuture, SyftFuture

__Q_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS futures (
    id TEXT PRIMARY KEY,
    bid TEXT DEFAULT NULL,
    path TEXT NOT NULL,
    expires TIMESTAMP NOT NULL,
    namespace TEXT NOT NULL
) WITHOUT ROWID
"""

__Q_INSERT_FUTURE = """
INSERT OR REPLACE INTO futures (id, path, expires, namespace, bid)
VALUES (:id, :path, :expires, :namespace, :bid)
"""


thread_local = threading.local()


@cache
def get_default_client():
    return Client.load()


def __get_connection(client: Client) -> sqlite3.Connection:
    if not hasattr(thread_local, "conn"):
        db_dir = client.workspace.plugins
        db_dir.mkdir(exist_ok=True, parents=True)
        db_path = db_dir / "rpc.futures.db"
        conn = sqlite3.connect(str(db_path))

        # Multi-process optimizations for small writes
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and speed
        conn.execute("PRAGMA cache_size=-2000")  # 2MB cache
        conn.execute("PRAGMA busy_timeout=5000")  # Wait up to 5s on locks
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA foreign_keys=OFF")

        conn.row_factory = sqlite3.Row

        conn.execute(__Q_CREATE_TABLE)
        conn.commit()
        thread_local.conn = conn

    return thread_local.conn


def save_future(
    future: SyftFuture,
    namespace: str,
    client: Optional[Client] = None,
    bulk_id: Optional[str] = None,
) -> str:
    client = client or get_default_client()
    conn = __get_connection(client)
    data = future.model_dump(mode="json")

    conn.execute(__Q_INSERT_FUTURE, {**data, "namespace": namespace, "bid": bulk_id})
    conn.commit()

    return data["id"]


def get_future(
    future_id: Union[UUID, str], client: Optional[Client] = None
) -> Optional[SyftFuture]:
    client = client or get_default_client()
    conn = __get_connection(client)
    row = conn.execute(
        "SELECT id, path, expires FROM futures WHERE id = ?", (str(future_id),)
    ).fetchone()

    if not row:
        return None

    return SyftFuture(**dict(row))


def delete_future(future_id: Union[UUID, str], client: Optional[Client] = None) -> None:
    client = client or get_default_client()
    conn = __get_connection(client)
    conn.execute("DELETE FROM futures WHERE id = ?", (str(future_id),))
    conn.commit()


def cleanup_expired_futures(client: Optional[Client] = None) -> None:
    client = client or Client.load()
    conn = __get_connection(client)
    conn.execute("DELETE FROM futures WHERE expires < datetime('now')")
    conn.commit()


def list_futures(namespace: Optional[str] = None, client: Optional[Client] = None):
    client = client or Client.load()
    conn = __get_connection(client)
    query_all = "SELECT id, path, expires FROM futures"
    query_app = "SELECT id, path, expires FROM futures WHERE namespace = ?"

    if namespace:
        rows = conn.execute(query_app, (namespace,)).fetchall()
    else:
        rows = conn.execute(query_all).fetchall()
    return [SyftFuture(**dict(row)) for row in rows]


def save_bulk_future(
    bulk_future: SyftBulkFuture,
    namespace: str,
    client: Optional[Client] = None,
) -> str:
    bid = str(bulk_future.id)
    for future in bulk_future.futures:
        save_future(future, namespace, client, bid)
    return bid


def get_bulk_future(
    bulk_id: Union[str, UUID], client: Optional[Client] = None
) -> Optional[SyftBulkFuture]:
    client = client or get_default_client()
    conn = __get_connection(client)
    rows = conn.execute(
        "SELECT id, path, expires FROM futures WHERE bid = ? ORDER BY expires",
        (str(bulk_id),),
    ).fetchall()

    if not rows:
        return None

    futures = [SyftFuture(**dict(row)) for row in rows]
    return SyftBulkFuture(futures=futures)


def delete_bulk_future(
    bulk_id: Union[str, UUID], client: Optional[Client] = None
) -> None:
    client = client or get_default_client()
    conn = __get_connection(client)
    conn.execute("DELETE FROM futures WHERE bid = ?", (str(bulk_id),))
    conn.commit()
