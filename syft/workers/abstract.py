from abc import ABC
from syft.interfaces.msgpack_interface import MsgpackInterface


class AbstractWorker(ABC, MsgpackInterface):
    pass
