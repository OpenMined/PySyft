from __future__ import annotations
from ....decorators import syft_decorator
from ...message import SyftMessage
from ....common import AbstractWorker
from ....common.token import Token
from typing import List


class WorkerService:
    authorization_payload = {}

    @staticmethod
    @syft_decorator(typechecking=True)
    def process(worker: AbstractWorker, msg: SyftMessage) -> object:
        self.authorization_payload = self._decode_msg_token(msg)


    @staticmethod
    @syft_decorator(typechecking=True)
    def message_handler_types() -> List[type]:
        raise NotImplementedError

    def _decode_msg_token(self, msg):
        if not msg.token:
            return {}
        rerturn Token().decode(token = msg.token, secret = 'THIS_SHOULD_BE_ENV_VARIABLE')
