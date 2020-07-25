from .specs.provider import Provider
from .specs.storage import Storage
from .specs.cpu import CPU
from .specs.gpu import GPU
from .specs.network import Network


class DeviceType:
    def __init__(
        self,
        name: str,
        provider: Provider,
        price_per_hour: float,
        memory: int,
        bare_metal: bool,
        storage: Storage,
        cpu: CPU,
        gpu: GPU,
        network: Network,
        spot_mode_supported: bool = False,
        price_per_hour_spot: float = 0,
    ):
        self.name = name
        self.provider = provider
        self.price_per_hour = price_per_hour
        self.memory = memory
        self.bare_metal = bare_metal
        self.storage = storage
        self.cpu = cpu
        self.gpu = gpu
        self.network = network
        self.spot_mode_supported = spot_mode_supported
        self.price_per_hour = price_per_hour_spot
