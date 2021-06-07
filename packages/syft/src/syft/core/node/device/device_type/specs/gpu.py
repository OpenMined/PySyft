# stdlib
from dataclasses import dataclass


@dataclass
class GPU:
    name: str
    count: int
    manufacturer: str
    memory_total: int
    memory_per_gpu: int
