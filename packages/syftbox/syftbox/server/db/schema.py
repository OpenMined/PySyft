import sqlite3
from pathlib import Path

from syftbox.lib.types import PathLike


# @contextlib.contextmanager
def get_db(path: PathLike) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(path), check_same_thread=False)

    with conn:
        conn.execute("PRAGMA cache_size=10000;")
        conn.execute("PRAGMA synchronous=OFF;")
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row

        # Create the table if it doesn't exist
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS file_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            datasite TEXT NOT NULL,
            path TEXT NOT NULL UNIQUE,
            hash TEXT NOT NULL,
            signature TEXT NOT NULL,
            file_size INTEGER NOT NULL,
            last_modified TEXT NOT NULL        )
        """
        )
        # TODO: migrate file_metadata id?

        # Create a table for storing file information
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rules (
                permfile_path varchar(1000) NOT NULL,
                permfile_dir varchar(1000) NOT NULL,
                permfile_depth INTEGER NOT NULL,
                priority INTEGER NOT NULL,
                path varchar(1000) NOT NULL,
                user varchar(1000) NOT NULL,
                can_read bool NOT NULL,
                can_create bool NOT NULL,
                can_write bool NOT NULL,
                admin bool NOT NULL,
                disallow bool NOT NULL,
                PRIMARY KEY (permfile_path, priority)
            )
        """
        )

        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS rule_files (
            permfile_path varchar(1000) NOT NULL,
            priority INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            match_for_email varchar(1000),
            PRIMARY KEY (permfile_path, priority, file_id),
            FOREIGN KEY (permfile_path, priority) REFERENCES rules(permfile_path, priority) ON DELETE CASCADE,
            FOREIGN KEY (file_id) REFERENCES file_metadata(id) ON DELETE CASCADE
        );
        """
        )
    return conn
