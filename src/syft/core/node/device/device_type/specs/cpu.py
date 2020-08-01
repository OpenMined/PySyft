from enum import Enum


class CpuArchitectureTypes(Enum):
    x86_64 = 0


class CPU:
    def __init__(
        self,
        num_cores: int,
        num_threads_per_core: int,
        clock_speed: float,
        architecture: CpuArchitectureTypes,
    ):
        self.num_cores = num_cores
        self.num_threads_per_core = num_threads_per_core
        self.clock_speed = clock_speed
        self.architecture = architecture
