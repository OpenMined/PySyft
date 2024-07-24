# third party
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

# relative
from ..abstract_server import ServerSideType
from ..abstract_server import ServerType


class UvicornSettings(BaseSettings):
    name: str
    server_type: ServerType = ServerType.DATASITE
    server_side_type: ServerSideType = ServerSideType.HIGH_SIDE
    processes: int = 1
    reset: bool = False
    dev_mode: bool = False
    enable_warnings: bool = False
    in_memory_workers: bool = True
    queue_port: int | None = None
    create_producer: bool = False
    n_consumers: int = 0
    association_request_auto_approval: bool = False
    background_tasks: bool = False

    # Profiling inputs
    profile: bool = False
    profile_interval: float = 0.001
    profile_dir: str | None = None

    model_config = SettingsConfigDict(env_prefix="SYFT_", env_parse_none_str="None")
