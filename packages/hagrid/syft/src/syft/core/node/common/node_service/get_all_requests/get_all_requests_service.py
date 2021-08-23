# # stdlib
# from typing import List
# from typing import Optional
#
# # third party
# from nacl.signing import VerifyKey
#
# # relative
# from ......logger import traceback_and_raise
# from ....abstract.node import AbstractNode
# from ....common.node_service.node_service import ImmediateNodeServiceWithoutReply
# from ..request_receiver.request_receiver_messages import RequestMessage
# from .get_all_requests_messages import GetAllRequestsMessage
# from .get_all_requests_messages import GetAllRequestsResponseMessage
#
#
# class GetAllRequestsService(ImmediateNodeServiceWithoutReply):
#     @staticmethod
#     def message_handler_types() -> List[type]:
#         return [GetAllRequestsMessage]
#
#     @staticmethod
#     def process(
#         node: AbstractNode,
#         msg: GetAllRequestsMessage,
#         verify_key: Optional[VerifyKey] = None,
#     ) -> GetAllRequestsResponseMessage:
#         try:
#             if verify_key is None:
#                 traceback_and_raise(
#                     ValueError(
#                         "Can't process Request service without a given " "verification key"
#                     )
#                 )
#
#             if verify_key == node.root_verify_key:
#                 return GetAllRequestsResponseMessage(
#                     requests=node.requests, address=msg.reply_to
#                 )
#
#             # only return requests which concern the user asking
#             valid_requests: List[RequestMessage] = list()
#             for request in node.requests:
#                 if request.requester_verify_key == verify_key:
#                     valid_requests.append(request)
#
#             return GetAllRequestsResponseMessage(
#                 requests=valid_requests, address=msg.reply_to
#             )
#         except Exception as e:
#             print('\n\nSOMETHING WENT WRONG!!!\n\n')
#             print(e)
