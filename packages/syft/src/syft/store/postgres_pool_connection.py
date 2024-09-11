# stdlib
from collections.abc import Generator
from contextlib import contextmanager
import logging
import time

# third party
from psycopg import Connection
from psycopg_pool import ConnectionPool
from psycopg_pool import PoolTimeout

# relative
from ..types.errors import SyftException

logger = logging.getLogger(__name__)


MIN_DB_POOL_SIZE = 1
MAX_DB_POOL_SIZE = 10
DEFAULT_POOL_CONN_TIMEOUT = 30
CONN_RETRY_INTERVAL = 1


class PostgresPoolConnection:
    def __init__(
        self,
        client_config: dict,
        min_size: int = MIN_DB_POOL_SIZE,
        max_size: int = MAX_DB_POOL_SIZE,
        timeout: int = DEFAULT_POOL_CONN_TIMEOUT,
        retry_interval: int = CONN_RETRY_INTERVAL,
        pool_kwargs: dict | None = None,
    ) -> None:
        connect_kwargs = self._connection_kwargs_from_config(client_config)

        # https://www.psycopg.org/psycopg3/docs/advanced/prepare.html#using-prepared-statements-with-pgbouncer
        # This should default to None to allow the connection pool to manage the prepare threshold
        connect_kwargs["prepare_threshold"] = None

        self.pool = ConnectionPool(
            kwargs=connect_kwargs,
            open=False,
            check=ConnectionPool.check_connection,
            min_size=min_size,
            max_size=max_size,
            **pool_kwargs,
        )
        logger.info(
            f"Connection pool created with min_size={self.min_size} and max_size={self.max_size}"
        )
        logger.info(f"Connected to {self.store_config.client_config.dbname}")
        logger.info(f"PostgreSQL Pool connection: {self.pool.get_stats()}")
        self.timeout = timeout
        self.retry_interval = retry_interval

    @contextmanager
    def get_connection(self) -> Generator[Connection, None, None]:
        """Provide a connection from the pool, waiting if necessary until one is available."""
        conn = None
        start_time = time.time()

        try:
            while True:
                try:
                    conn = self.pool.getconn(timeout=self.retry_interval)
                    if conn:
                        yield conn  # Return the connection object to be used in the context
                        break
                except PoolTimeout as e:
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= self.timeout:
                        message = f"Could not get a connection from database pool within {self.timeout} seconds."
                        raise SyftException.from_exception(
                            e,
                            public_message=message,
                        )
                    logger.warning(
                        f"Connection not available, retrying... ({elapsed_time:.2f} seconds elapsed)"
                    )
                    time.sleep(self.retry_interval)

        except Exception as e:
            logger.error(f"Error getting connection from pool: {e}")
            yield None
        finally:
            if conn:
                self.pool.putconn(conn)

    def release_connection(self, conn: Connection) -> None:
        """Release a connection back to the pool."""
        try:
            if conn.closed or conn.broken:
                self.pool.putconn(conn, close=True)
                logger.info("Broken connection closed and removed from pool.")
            else:
                self.pool.putconn(conn)
                logger.info("Connection released back to pool.")
        except Exception as e:
            logger.error(f"Error releasing connection: {e}")

    def _connection_kwargs_from_config(self, config: dict) -> dict:
        return {
            "dbname": config.get("dbname"),
            "user": config.get("user"),
            "password": config.get("password"),
            "host": config.get("host"),
            "port": config.get("port"),
        }

    def close_all_connections(self) -> None:
        """Close all connections in the pool and shut down the pool."""
        try:
            self.pool.close()
            logger.info("All connections closed and pool shut down.")
        except Exception as e:
            logger.error(f"Error closing connection pool: {e}")
