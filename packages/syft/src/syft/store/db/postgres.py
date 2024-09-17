# third party
from sqlalchemy import URL

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from .db import DBManager
from .sqlite import DBConfig


@serializable(canonical_name="PostgresDBConfig", version=1)
class PostgresDBConfig(DBConfig):
    host: str
    port: int
    user: str
    password: str
    database: str

    @property
    def connection_string(self) -> str:
        return URL.create(
            "postgresql",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=self.database,
        ).render_as_string(hide_password=False)


class PostgresDBManager(DBManager[PostgresDBConfig]):
    def update_settings(self) -> None:
        return super().update_settings()

    @classmethod
    def random(
        cls: type,
        *,
        config: PostgresDBConfig,
        server_uid: UID | None = None,
        root_verify_key: SyftVerifyKey | None = None,
    ) -> "PostgresDBManager":
        root_verify_key = root_verify_key or SyftVerifyKey.generate()
        server_uid = server_uid or UID()
        return PostgresDBManager(
            config=config, server_uid=server_uid, root_verify_key=root_verify_key
        )
