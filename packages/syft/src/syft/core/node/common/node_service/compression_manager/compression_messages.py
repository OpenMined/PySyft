# stdlib
from enum import Enum
from typing import List
from typing import Optional

# third party
from google.protobuf.reflection import GeneratedProtocolMessageType
from nacl.signing import VerifyKey
import time

# relative
from .....common.serde.deserialize import _deserialize
from ...... import serialize
from ......logger import critical
from ......logger import debug
from ......logger import traceback
from ......logger import traceback_and_raise
from ......proto.core.node.domain.service.compression_params_message_pb2 import (
    CompressionBytesMessage as CompressionBytesMessage_PB,
)
from ......proto.core.node.domain.service.compression_params_message_pb2 import (
    CompressionConnectionMessage as CompressionConnectionMessage_PB,
)
from ......proto.core.node.domain.service.compression_params_message_pb2 import (
    CompressionDeepReduceMessage as CompressionDeepReduceMessage_PB,
)
from ......proto.core.node.domain.service.compression_params_message_pb2 import (
    CompressionParamsMessage as CompressionParamsMessage_PB,
)
from ......proto.core.node.domain.service.compression_params_message_pb2 import (
    CompressionTensorMessage as CompressionTensorMessage_PB,
)
from ......proto.core.node.domain.service.compression_params_message_pb2 import (
    CompressionDGCCompressorMessage as CompressionDGCCompressorMessage_PB,
)
from .....common import UID
from .....common.message import ImmediateSyftMessageWithReply
from .....common.serde.serializable import bind_protobuf
from .....io.address import Address
from .....compression.compression_params import compression_params
from .....compression.compression_params import CompressionParams 

@bind_protobuf
class CompressionParamsMessage(ImmediateSyftMessageWithReply):

    def __init__(
        self,
        address: Address,
        reply_to: Address,
        msg_id: Optional[UID] = None,
        status = 'update',
        compression_params = CompressionParams(),
        time = time.time(),
    ):
        super().__init__(address=address, msg_id=msg_id, reply_to=reply_to)
        self.status = status
        self.compression_params = compression_params
        self.time = time

    def _object2proto(self) -> CompressionParamsMessage_PB:
        msg = CompressionParamsMessage_PB()
        msg.bytes.CopyFrom(
            CompressionBytesMessage_PB(
                compress=self.compression_params.bytes['compress'],
                lib=self.compression_params.bytes['lib'],
                cname=self.compression_params.bytes['cname'],
                compression_lvl=self.compression_params.bytes['compression_lvl'],
            )
        )
        msg.tensor.CopyFrom(
            CompressionTensorMessage_PB(
                compress=self.compression_params.tensor['compress'],
                compressors=self.compression_params.tensor['compressors'],
            )
        )
        msg.dgc_compressor.CopyFrom(
            CompressionDGCCompressorMessage_PB(
                ratio=self.compression_params.dgc_compressor['ratio'],
            )
        )
        msg.deep_reduce.CopyFrom(
            CompressionDeepReduceMessage_PB(
                compress_ratio=self.compression_params.deep_reduce['compress_ratio'],
                deepreduce=self.compression_params.deep_reduce['deepreduce'],
                index=self.compression_params.deep_reduce['index'],
            )
        )
        msg.connection.CopyFrom(
            CompressionConnectionMessage_PB(
                speed=self.compression_params.connection['speed'],
                tested=self.compression_params.connection['tested'],
            )
        )
        msg.reply_to.CopyFrom(
            serialize(self.reply_to)
        )
        msg.address.CopyFrom(
            serialize(self.address)
        )
        msg.msg_id.CopyFrom(
            serialize(self.id)
        )
        msg.status = self.status
        msg.time = self.time
        return msg

    @staticmethod
    def _proto2object(proto: CompressionParamsMessage_PB) -> "CompressionParamsMessage":
        bytes = {
            'compress': proto.bytes.compress,
            'lib': proto.bytes.lib,
            'cname': proto.bytes.cname,
            'compression_lvl': proto.bytes.compression_lvl,
        }
        tensor = {
            'compress': proto.tensor.compress,
            'compressors': proto.tensor.compressors,
        }
        dgc_compressor = {
            'ratio': proto.dgc_compressor.ratio,
        }
        deep_reduce = {
            'compress_ratio': proto.deep_reduce.compress_ratio, 
            'deepreduce':proto.deep_reduce.deepreduce, 
            'index':proto.deep_reduce.index,
        }
        connection = {
            'speed': proto.connection.speed,
            'tested': proto.connection.tested,
        }
        res = CompressionParams()
        res._bytes = bytes
        res._tensor = tensor
        res._dgc_compressor = dgc_compressor
        res._deep_reduce = deep_reduce
        res._connection = connection
        res.fix_dict_listener()
        return CompressionParamsMessage(
            address=_deserialize(blob=proto.address),
            reply_to=_deserialize(blob=proto.reply_to),
            msg_id=_deserialize(blob=proto.msg_id),
            compression_params=res,
            status=proto.status,
            time=proto.time,
        )

    @staticmethod
    def get_protobuf_schema() -> GeneratedProtocolMessageType:
        return CompressionParamsMessage_PB
