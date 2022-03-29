# # stdlib
# from typing import Callable
# from typing import Dict
# from typing import List
# from typing import Type
# from typing import Union

# # third party
# from nacl.encoding import HexEncoder
# from nacl.signing import SigningKey
# from nacl.signing import VerifyKey
# import torch as th

# # relative
# from .....common.group import VerifyAll
# from .....common.message import ImmediateSyftMessageWithReply
# from .....common.message import SignedImmediateSyftMessageWithoutReply
# from .....common.uid import UID
# from .....store.storeable_object import StorableObject
# from ....abstract.node import AbstractNode
# from ...action.save_object_action import SaveObjectAction
# from ..auth import service_auth
# from ..node_service import ImmediateNodeServiceWithReply
# from .tensor_manager_messages import CreateTensorMessage
# from .tensor_manager_messages import CreateTensorResponse
# from .tensor_manager_messages import DeleteTensorMessage
# from .tensor_manager_messages import DeleteTensorResponse
# from .tensor_manager_messages import GetTensorMessage
# from .tensor_manager_messages import GetTensorResponse
# from .tensor_manager_messages import GetTensorsMessage
# from .tensor_manager_messages import GetTensorsResponse
# from .tensor_manager_messages import UpdateTensorMessage
# from .tensor_manager_messages import UpdateTensorResponse


# def create_tensor_msg(
#     msg: CreateTensorMessage,
#     node: AbstractNode,
# ) -> CreateTensorResponse:
#     try:
#         payload = msg.content

#         new_tensor = th.tensor(payload["tensor"])
#         new_tensor.tag(*payload.get("tags", []))
#         new_tensor.describe(payload.get("description", ""))

#         id_at_location = UID()

#         # Step 2: create message which contains object to send
#         storable = StorableObject(
#             id=id_at_location,
#             data=new_tensor,
#             tags=new_tensor.tags,
#             description=new_tensor.description,
#             search_permissions={VerifyAll(): None}
#             if payload.get("searchable", False)
#             else {},
#         )

#         obj_msg = SaveObjectAction(obj=storable, address=node.address)

#         signed_message: SignedImmediateSyftMessageWithoutReply = obj_msg.sign(
#             signing_key=SigningKey(
#                 payload["internal_key"].encode("utf-8"), encoder=HexEncoder
#             )
#         )

#         node.recv_immediate_msg_without_reply(msg=signed_message)

#         return CreateTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={
#                 "msg": "Tensor created succesfully!",
#                 "tensor_id": str(id_at_location.value),
#             },
#         )
#     except Exception as e:
#         return CreateTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={"error": str(e)},
#         )


# def update_tensor_msg(
#     msg: UpdateTensorMessage,
#     node: AbstractNode,
# ) -> UpdateTensorResponse:
#     try:
#         payload = msg.content

#         new_tensor = th.tensor(payload["tensor"])
#         new_tensor.tag(*payload.get("tags", []))
#         new_tensor.describe(payload.get("description", ""))

#         key = UID.from_string(value=payload["tensor_id"])

#         # Step 2: create message which contains object to send
#         storable = StorableObject(
#             id=key,
#             data=new_tensor,
#             tags=new_tensor.tags,
#             description=new_tensor.description,
#             search_permissions={VerifyAll(): None}
#             if payload.get("searchable", False)
#             else {},
#         )

#         obj_msg = SaveObjectAction(obj=storable, address=node.address)

#         signed_message: SignedImmediateSyftMessageWithoutReply = obj_msg.sign(
#             signing_key=SigningKey(
#                 payload["internal_key"].encode("utf-8"), encoder=HexEncoder
#             )
#         )

#         node.recv_immediate_msg_without_reply(msg=signed_message)

#         return UpdateTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={"msg": "Tensor modified successfully!"},
#         )
#     except Exception as e:
#         return UpdateTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={"error": str(e)},
#         )


# def get_tensor_msg(
#     msg: GetTensorMessage,
#     node: AbstractNode,
# ) -> GetTensorResponse:
#     try:
#         payload = msg.content

#         # Retrieve the dataset from node.store
#         key = UID.from_string(value=payload["tensor_id"])
#         tensor = node.store.get(key)
#         return GetTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={
#                 "tensor": {
#                     "id": payload["tensor_id"],
#                     "tags": tensor.tags,
#                     "description": tensor.description,
#                 }
#             },
#         )
#     except Exception as e:
#         return GetTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={"error": str(e)},
#         )


# def get_tensors_msg(
#     msg: GetTensorsMessage,
#     node: AbstractNode,
# ) -> GetTensorsResponse:
#     try:
#         tensors = node.store.get_objects_of_type(obj_type=th.Tensor)

#         result = []

#         for tensor in tensors:
#             result.append(
#                 {
#                     "id": str(tensor.id.value),
#                     "tags": tensor.tags,
#                     "description": tensor.description,
#                 }
#             )
#         return GetTensorsResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={"tensors": result},
#         )
#     except Exception as e:
#         return GetTensorsResponse(
#             address=msg.reply_to, content={"error": str(e)}, status_code=200
#         )


# def del_tensor_msg(
#     msg: DeleteTensorMessage,
#     node: AbstractNode,
# ) -> DeleteTensorResponse:
#     try:
#         payload = msg.content

#         # Retrieve the dataset from node.store
#         key = UID.from_string(value=payload["tensor_id"])
#         node.store.delete(key=key)

#         return DeleteTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={"msg": "Tensor deleted successfully!"},
#         )
#     except Exception as e:
#         return DeleteTensorResponse(
#             address=msg.reply_to,
#             status_code=200,
#             content={"error": str(e)},
#         )


# class TensorManagerService(ImmediateNodeServiceWithReply):
#     INPUT_TYPE = Union[
#         Type[CreateTensorMessage],
#         Type[UpdateTensorMessage],
#         Type[GetTensorMessage],
#         Type[GetTensorsMessage],
#         Type[DeleteTensorMessage],
#     ]

#     INPUT_MESSAGES = Union[
#         CreateTensorMessage,
#         UpdateTensorMessage,
#         GetTensorMessage,
#         GetTensorsMessage,
#         DeleteTensorMessage,
#     ]

#     OUTPUT_MESSAGES = Union[
#         CreateTensorResponse,
#         UpdateTensorResponse,
#         GetTensorResponse,
#         GetTensorsResponse,
#         DeleteTensorResponse,
#     ]

#     msg_handler_map: Dict[INPUT_TYPE, Callable[..., OUTPUT_MESSAGES]] = {
#         CreateTensorMessage: create_tensor_msg,
#         UpdateTensorMessage: update_tensor_msg,
#         GetTensorMessage: get_tensor_msg,
#         GetTensorsMessage: get_tensors_msg,
#         DeleteTensorMessage: del_tensor_msg,
#     }

#     @staticmethod
#     @service_auth(guests_welcome=True)
#     def process(
#         node: AbstractNode,
#         msg: INPUT_MESSAGES,
#         verify_key: VerifyKey,
#     ) -> OUTPUT_MESSAGES:
#         return TensorManagerService.msg_handler_map[type(msg)](msg=msg, node=node)

#     @staticmethod
#     def message_handler_types() -> List[Type[ImmediateSyftMessageWithReply]]:
#         return [
#             CreateTensorMessage,
#             UpdateTensorMessage,
#             GetTensorMessage,
#             GetTensorsMessage,
#             DeleteTensorMessage,
#         ]
