# stdlib
from dataclasses import dataclass
from enum import Enum


class CpuArchitectureTypes(Enum):
    x86_64 = 0


@dataclass
class CPU:
    num_cores: int
    num_threads_per_core: int
    clock_speed: float
    architecture: CpuArchitectureTypes
