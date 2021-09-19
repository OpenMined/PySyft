# stdlib
import time
from typing import Any
from typing import Dict as DictType
from typing import List
from typing import Optional

# third party
from nacl.signing import VerifyKey

# relative
from ......logger import debug
from ......logger import error
from ......util import traceback_and_raise
from ....abstract.node import AbstractNode
from ..node_service import ImmediateNodeServiceWithoutReply
from .request_handler_messages import GetAllRequestHandlersMessage
from .request_handler_messages import GetAllRequestHandlersResponseMessage
from .request_handler_messages import UpdateRequestHandlerMessage


class UpdateRequestHandlerService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [UpdateRequestHandlerMessage]

    @staticmethod
    def process(
        node: AbstractNode,
        msg: UpdateRequestHandlerMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> None:
        if verify_key is None:
            traceback_and_raise(
                ValueError(
                    "Can't process Request service without a given " "verification key"
                )
            )
        if verify_key == node.root_verify_key:
            replacement_handlers = []

            # find if there exists a handler match the handler passed in
            existing_handlers = getattr(node, "request_handlers", None)
            debug(f"> Updating Request Handlers with existing: {existing_handlers}")
            if existing_handlers is not None:
                matched = None
                for existing_handler in existing_handlers:
                    # we match two handlers according to their tags
                    if existing_handler["tags"] == msg.handler["tags"]:
                        matched = existing_handler
                        # if an existing_handler match the passed in handler,
                        # we ignore it in for loop
                        continue
                    else:
                        # if an existing_handler does not match the passed in
                        # handler, we keep it
                        replacement_handlers.append(existing_handler)

                if msg.keep:
                    msg.handler["created_time"] = time.time()
                    replacement_handlers.append(msg.handler)
                    if matched is not None:
                        debug(
                            f"> Replacing a Request Handler {matched} with: {msg.handler}"
                        )
                    else:
                        debug(f"> Adding a Request Handler {msg.handler}")
                else:
                    debug(f"> Removing a Request Handler with: {msg.handler}")

                setattr(node, "request_handlers", replacement_handlers)
                debug(f"> Finished Updating Request Handlers with: {existing_handlers}")
            else:
                error(f"> Node has no Request Handlers attribute: {type(node)}")

        return


class GetAllRequestHandlersService(ImmediateNodeServiceWithoutReply):
    @staticmethod
    def message_handler_types() -> List[type]:
        return [GetAllRequestHandlersMessage]

    @staticmethod
    def process(
        node: AbstractNode,
        msg: GetAllRequestHandlersMessage,
        verify_key: Optional[VerifyKey] = None,
    ) -> GetAllRequestHandlersResponseMessage:

        if verify_key is None:
            traceback_and_raise(
                ValueError(
                    "Can't process Request service without a given " "verification key"
                )
            )

        handlers: List[DictType[str, Any]] = []
        if verify_key == node.root_verify_key:
            existing_handlers = getattr(node, "request_handlers", None)
            debug(f"> Getting all Existing Request Handlers: {existing_handlers}")
            if existing_handlers is not None:
                handlers = existing_handlers

        return GetAllRequestHandlersResponseMessage(
            handlers=handlers, address=msg.reply_to
        )
