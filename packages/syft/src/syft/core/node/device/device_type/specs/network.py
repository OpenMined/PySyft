# stdlib
from dataclasses import dataclass


@dataclass
class Network:
    performance: str
    ipv6_supported: bool = False
