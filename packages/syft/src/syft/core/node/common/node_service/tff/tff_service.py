# stdlib
from typing import List
from typing import Optional
from typing import Type

# third party
from nacl.signing import VerifyKey
import tensorflow_federated as tff

# relative
from ......util import traceback_and_raise
from ....abstract.node import AbstractNode
from ..auth import service_auth
from ..node_service import ImmediateNodeServiceWithReply
from ..node_service import EventualNodeServiceWithoutReply
from .tff_messages import TFFMessage
from .tff_messages import TFFReplyMessage

async def support():
    string = await tff.federated_computation(lambda: 'Hello World')()
    print("in async context", string)
    return string


class TFFService(ImmediateNodeServiceWithReply):
    @staticmethod
    @service_auth(guests_welcome=True)
    def process(
        node: AbstractNode, msg: TFFMessage, verify_key: Optional[VerifyKey] = None
    ) -> TFFReplyMessage:
        if verify_key is None:
            traceback_and_raise("Can't process TFFService with no verification key.")
        # import nest_asyncio
        import asyncio
        # import uvloop
        # nest_asyncio.apply()
        # asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        # print(asyncio.get_event_loop_policy())
        # print(dir(asyncio.get_event_loop_policy()))
        tff.backends.native.execution_contexts.set_local_async_python_execution_context(reference_resolving_clients=True)
        loop = asyncio.get_event_loop()
        
        loop.run_until_complete(asyncio.wait([loop.create_task(support())]))
        loop.close()
        
        
        # s = 0
        # for x in range(100000000):
        #     s += x
        # print(s)
        # loop = asyncio.get_running_loop()
        # loop.run_until_complete(lambda: await tff.federated_computation(lambda: 'Hello World')())
        # print(res)

        result = msg.payload.run(node=node, verify_key=verify_key)
        return TFFReplyMessage(payload=result, address=msg.reply_to)

    @staticmethod
    def message_handler_types() -> List[Type[TFFMessage]]:
        return [TFFMessage]
