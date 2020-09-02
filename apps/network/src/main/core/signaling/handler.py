from typing import Optional, Union
from syft.core.common.serde.serializable import Serializable

from syft.core.io.address import Address

from syft.grid.services.signaling_service import (
    SignalingOfferMessage,
    SignalingAnswerMessage,
    OfferPullRequestMessage,
    AnswerPullRequestMessage,
)

from ...storage.memory_storage import MemoryStorage


class SignalingHandler(object):
    def __init__(self):
        self.offer_msgs = MemoryStorage()
        self.answer_msgs = MemoryStorage()

    def push(self, msg: Union[SignalingOfferMessage, SignalingAnswerMessage]) -> None:
        if isinstance(msg, SignalingOfferMessage):
            _map = self.offer_msgs
        elif isinstance(msg, SignalingAnswerMessage):
            _map = self.answer_msgs

        addr_map = _map.get(msg.address.name)

        if addr_map:
            addr_map[msg.reply_to] = msg
        else:
            _map.register(key=msg.address.name, value={msg.reply_to.name: msg})

    def pull(
        self, msg: Union[OfferPullRequestMessage, AnswerPullRequestMessage]
    ) -> Union[SignalingOfferMessage, SignalingAnswerMessage]:
        if isinstance(msg, OfferPullRequestMessage):
            return self._consume(msg=msg, queue=self.offer_msgs)

        elif isinstance(msg, AnswerPullRequestMessage):
            return self._consume(msg=msg, queue=self.answer_msgs)

        return None

    def _consume(
        self,
        msg: Union[OfferPullRequestMessage, AnswerPullRequestMessage],
        queue: MemoryStorage,
    ) -> Union[SignalingOfferMessage, None]:
        _addr_map = queue.get(msg.reply_to.name)

        if _addr_map:
            return _addr_map.get(msg.address.name, None)
