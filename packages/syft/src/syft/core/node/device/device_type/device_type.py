# stdlib
from dataclasses import dataclass
from typing import Optional

# syft relative
from .specs.cpu import CPU
from .specs.gpu import GPU
from .specs.network import Network
from .specs.provider import Provider
from .specs.storage import Storage


@dataclass
class DeviceType:
    name: str
    provider: Provider
    price_per_hour: float
    memory: int
    bare_metal: bool
    storage: Storage
    cpu: CPU
    gpu: Optional[GPU]
    network: Optional[Network]
    spot_mode_supported: bool = False
    price_per_hour_spot: float = 0
