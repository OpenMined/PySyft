from typing import Optional, Union
from syft.core.common.serde.serializable import Serializable
from syft.core.common.serde.deserialize import _deserialize
from syft.core.io.address import Address

from syft.grid.services.signaling_service import (
    SignalingOfferMessage,
    SignalingAnswerMessage,
)


class SignalingHandler(object):
    def __init__(self):
        self.offer_msgs = dict()
        self.answer_msgs = dict()

    def store_msg(self, msg: Serializable, bidirectional_conn: bool = False) -> None:
        _msg = _deserialize(blob=msg, from_json=True)

        if isinstance(_msg, SignalingOfferMessage):
            _queue = self.offer_msgs
        elif isinstance(_msg, SignalingAnswerMessage):
            _queue = self.answer_msgs
        try:
            _queue[_msg.address].append(_msg)
        except KeyError:
            _queue[_msg.address] = [_msg]

    def consume_offer(self, addr: Address) -> Union[SignalingOfferMessage, None]:
        _addr = _deserialize(blob=addr, from_json=True)

        if _addr in self.offer_msgs and len(self.offer_msgs[_addr]) > 0:
            return self.offer_msgs[_addr].pop(0)

    def consume_answer(self, addr: Address) -> Union[SignalingAnswerMessage, None]:
        _addr = _deserialize(blob=addr, from_json=True)

        if _addr in self.answer_msgs and len(self.answer_msgs[_addr]) > 0:
            return self.answer_msgs[_addr].pop(0)
