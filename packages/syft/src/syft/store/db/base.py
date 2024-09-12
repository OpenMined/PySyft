# stdlib
import logging

# third party
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID

logger = logging.getLogger(__name__)


@serializable(canonical_name="DBConfig", version=1)
class DBConfig(BaseModel):
    reset: bool = False

    @property
    def connection_string(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class DBManager:
    def __init__(
        self,
        config: DBConfig,
        server_uid: UID,
        root_verify_key: SyftVerifyKey,
    ) -> None:
        self.config = config
        self.root_verify_key = root_verify_key
        self.server_uid = server_uid
        self.engine = create_engine(
            config.connection_string,
            # json_serializer=dumps,
            # json_deserializer=loads,
        )
        logger.info(f"Connecting to {config.connection_string}")
        self.sessionmaker = sessionmaker(bind=self.engine)
        logger.info(f"Successfully connected to {config.connection_string}")
        self.update_settings()

    def update_settings(self) -> None:
        pass

    def init_tables(self) -> None:
        pass

    def reset(self) -> None:
        pass
