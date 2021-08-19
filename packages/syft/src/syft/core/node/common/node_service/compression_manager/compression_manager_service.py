# stdlib
import time
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import traceback_and_raise
from .....node.common.node import DuplicateRequestException
from ....abstract.node import AbstractNode
from ....common.node_service.node_service import ImmediateNodeServiceWithReply
from .compression_messages import CompressionParamsMessage
from .....compression.compression_params import compression_params
from ..success_resp_message import SuccessResponseMessage

class CompressionParamsService(ImmediateNodeServiceWithReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [CompressionParamsMessage]

    @staticmethod
    def process(
        node: AbstractNode, msg: CompressionParamsMessage, verify_key: Optional[VerifyKey] = None
    ) -> None:
        print(msg.status)
        if msg.status == 'connection_testing':
            msg.status = 'connection_testing_resp'
            msg.time = time.time()
            return msg
        else:
            print('entered setter', compression_params._bytes, msg.compression_params.bytes)
            res = msg.compression_params
            compression_params._bytes = res.bytes
            compression_params._tensor = res.tensor
            compression_params._dgc_compressor = res.dgc_compressor
            compression_params._deep_reduce = res.deep_reduce
            compression_params._connection = res.connection
            return SuccessResponseMessage(
                address=msg.reply_to,
                resp_msg="Compression params updated successfully!",
            )


