# stdlib
import logging
from typing import Any

# third party
import psycopg
from psycopg import Cursor

# relative
from .postgres_pool_connection import PostgresPoolConnection

logger = logging.getLogger(__name__)


MAX_QUERY_RETRIES = 3
QUERY_RETRY_DELAY = 5


class PostgresQueryExecutor:
    def __init__(
        self,
        connection_pool: PostgresPoolConnection,
        retries: int = MAX_QUERY_RETRIES,
        retry_delay: int = QUERY_RETRY_DELAY,
    ) -> None:
        self.connection_pool = connection_pool
        self.retries = retries
        self.retry_delay = retry_delay

    def execute_query(self, query: str, args: list[Any] | None = None) -> Cursor | None:
        """
        Execute a query on the database using a context-managed connection.
        Handles `InFailedSqlTransaction` errors by rolling back the transaction.
        Returns a cursor object after execution for further handling by the caller.

        :param query: SQL query to execute.
        :param params: Query parameters (optional).
        :return: Cursor object or None if an error occurs.
        """
        attempt = 0
        while attempt < self.retries:
            try:
                # Using the context manager for the connection
                with self.connection_pool.get_connection() as conn:
                    if conn is None:
                        return None

                    cur = conn.cursor()

                    # Check if connection is in failed state (i.e., in a failed transaction)
                    if conn.status == psycopg.extensions.STATUS_IN_FAILED_TRANSACTION:
                        logger.warning(
                            "Transaction is in a failed state. Rolling back."
                        )
                        conn.rollback()

                    cur.execute(query, args)

                    conn.commit()

                    return cur  # Return the cursor object

            except psycopg.errors.InFailedSqlTransaction as e:
                logger.error(f"Transaction failed and is in an invalid state: {e}")
                if conn and not conn.closed:
                    conn.rollback()  # Roll back the transaction
                attempt += 1  # Retry the query after rollback

            except (psycopg.OperationalError, psycopg.errors.AdminShutdown) as e:
                logger.error(
                    f"Server error or termination: {e}. Retrying ({attempt + 1}/{self.retries})..."
                )
                attempt += 1

            except Exception as e:
                logger.error(f"Error executing query: {e}")
                if conn and not conn.closed:
                    conn.rollback()  # Roll back on any general error
                return None

        logger.error(f"Query failed after {self.retries} attempts.")
        return None
