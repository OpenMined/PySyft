from abc import ABC
from syft.serde.syft_serializable import SyftSerializable


class AbstractWorker(ABC, SyftSerializable):
    pass
