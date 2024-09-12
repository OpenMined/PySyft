# stdlib
import logging
from typing import Generic
from typing import TypeVar

# third party
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# relative
from ...serde.serializable import serializable
from ...server.credentials import SyftVerifyKey
from ...types.uid import UID
from .schema import Base

logger = logging.getLogger(__name__)


@serializable(canonical_name="DBConfig", version=1)
class DBConfig(BaseModel):
    reset: bool = False

    @property
    def connection_string(self) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


ConfigT = TypeVar("ConfigT", bound=DBConfig)


class DBManager(Generic[ConfigT]):
    def __init__(
        self,
        config: ConfigT,
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
        self.update_settings()
        logger.info(f"Successfully connected to {config.connection_string}")

    def update_settings(self) -> None:
        pass

    def init_tables(self) -> None:
        if self.config.reset:
            # drop all tables that we know about
            Base.metadata.drop_all(bind=self.engine)
            self.config.reset = False
        Base.metadata.create_all(self.engine)

    def reset(self) -> None:
        Base.metadata.drop_all(bind=self.engine)
        Base.metadata.create_all(self.engine)
