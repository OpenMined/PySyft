# future
from __future__ import annotations

# stdlib
import time
from typing import Any
from typing import Dict
from typing import Optional

# third party
from nacl.signing import VerifyKey
from typing_extensions import final

# relative
from .....common.serde.serializable import serializable
from ....abstract.node import AbstractNode
from ..generic_payload.messages import GenericPayloadMessage
from ..generic_payload.messages import GenericPayloadMessageWithReply
from ..generic_payload.messages import GenericPayloadReplyMessage


@serializable(recursive_serde=True)
@final
class SleepMessage(GenericPayloadMessage):
    ...


@serializable(recursive_serde=True)
@final
class SleepReplyMessage(GenericPayloadReplyMessage):
    ...


@serializable(recursive_serde=True)
@final
class SleepMessageWithReply(GenericPayloadMessageWithReply):
    message_type = SleepMessage
    message_reply_type = SleepReplyMessage

    def run(
        self, node: AbstractNode, verify_key: Optional[VerifyKey] = None
    ) -> Dict[str, Any]:
        seconds = float(self.kwargs["seconds"])
        time.sleep(seconds)
        return {
            "seconds": seconds,
            "result": f"Finished Sleeping for {seconds}",
            "status_code": 200,
        }
