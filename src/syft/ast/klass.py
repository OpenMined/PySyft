from typing import Tuple
from typing import Callable as CallableT
from typing import Union
from typing import Optional

from ..ast.callable import Callable
from ..core.pointer.pointer import Pointer
from ..core.node.common.action.run_class_method_action import RunClassMethodAction
from ..core.node.common.action.save_object_action import SaveObjectAction
from ..core.common.serde.serializable import Serializable
from ..core.common.serde.serialize import _serialize
from google.protobuf.message import Message
from ..util import aggressive_set_attr

# TODO: Fix circular import for Client interface
# from ..core.node.common.client import Client
from typing import Any


class Class(Callable):

    ""

    def __init__(
        self,
        name: Optional[str],
        path_and_name: Optional[str],
        ref: Union[Callable, CallableT],
        return_type_name: Optional[str],
    ):
        super().__init__(name, path_and_name, ref, return_type_name=return_type_name)
        if self.path_and_name is not None:
            self.pointer_name = self.path_and_name.split(".")[-1] + "Pointer"

    def __repr__(self) -> str:
        return f"<Class:{self.name}>"

    @property
    def pointer_type(self) -> Union[Callable, CallableT]:
        return getattr(self, self.pointer_name)

    def create_pointer_class(self) -> None:
        def run_class_method(
            attr: Union[Callable, CallableT],
            attr_path_and_name: str,
            __self: Any,
            args: Tuple[Any, ...],
            kwargs: Any,
        ) -> object:
            # TODO: lookup actual return type instead of just guessing that it's identical
            result = self.pointer_type(client=__self.client)

            # QUESTION can the id_at_location be None?
            result_id_at_location = getattr(result, "id_at_location", None)
            if result_id_at_location is not None:
                cmd = RunClassMethodAction(
                    path=attr_path_and_name,
                    _self=__self,
                    args=args,
                    kwargs=kwargs,
                    id_at_location=result_id_at_location,
                    address=__self.client.address,  # TODO: these uses of the word "location" should change to "address"
                )
                __self.client.send_immediate_msg_without_reply(msg=cmd)

            return result

        attrs = {}
        for attr_name, attr in self.attrs.items():
            attr_path_and_name = getattr(attr, "path_and_name", None)
            # QUESTION: Could path_and_name be None?
            # It seems as though attrs can contain
            # Union[Callable, CallableT]
            # where Callable is ast.callable.Callable
            # where CallableT is typing.Callable == any function, method, lambda
            # so we have to check for path_and_name
            if attr_path_and_name is not None:
                attrs[attr_name] = lambda _self, *args, **kwargs: run_class_method(
                    attr, attr_path_and_name, _self, args, kwargs
                )

        klass_pointer = type(self.pointer_name, (Pointer,), attrs)
        setattr(klass_pointer, "path_and_name", self.path_and_name)
        setattr(self, self.pointer_name, klass_pointer)

    def create_send_method(outer_self: Any) -> None:
        def send(self: Any, client: Any) -> Pointer:
            # Step 1: create pointer which will point to result
            ptr = getattr(outer_self, outer_self.pointer_name)(
                client=client,
                id_at_location=self.id,
                tags=self.tags if hasattr(self, "tags") else list(),
                description=self.description if hasattr(self, "description") else "",
            )

            # Step 2: create message which contains object to send
            obj_msg = SaveObjectAction(
                obj_id=ptr.id_at_location, obj=self, address=client.address
            )

            # Step 3: send message
            client.send_immediate_msg_without_reply(msg=obj_msg)

            # STep 4: return pointer
            return ptr

        # using curse because Numpy tries to lock down custom attributes
        aggressive_set_attr(obj=outer_self.ref, name="send", attr=send)

    def create_storable_object_attr_convenience_methods(outer_self: Any) -> None:
        def tag(self: Any, *tags: Tuple[Any, ...]) -> object:
            self.tags = list(tags)
            return self

        # using curse because Numpy tries to lock down custom attributes
        aggressive_set_attr(obj=outer_self.ref, name="tag", attr=tag)

        def describe(self: Any, description: str) -> object:
            self.description = description
            # QUESTION: Is this supposed to return self?
            # WHY? Chaining?
            return self

        # using curse because Numpy tries to lock down custom attributes
        aggressive_set_attr(obj=outer_self.ref, name="describe", attr=describe)

    def create_serialization_methods(outer_self) -> None:
        def serialize(  # type: ignore
            self,
            to_proto: bool = True,
            to_json: bool = False,
            to_binary: bool = False,
            to_hex: bool = False,
        ) -> Union[str, bytes, Message]:
            return _serialize(
                obj=self,
                to_proto=to_proto,
                to_json=to_json,
                to_binary=to_binary,
                to_hex=to_hex,
            )

        aggressive_set_attr(obj=outer_self.ref, name="serialize", attr=serialize)
        aggressive_set_attr(
            obj=outer_self.ref, name="to_proto", attr=Serializable.to_proto
        )
        aggressive_set_attr(obj=outer_self.ref, name="proto", attr=Serializable.proto)
        aggressive_set_attr(
            obj=outer_self.ref, name="to_json", attr=Serializable.to_json
        )
        aggressive_set_attr(obj=outer_self.ref, name="json", attr=Serializable.json)
        aggressive_set_attr(
            obj=outer_self.ref, name="to_binary", attr=Serializable.to_binary
        )
        aggressive_set_attr(obj=outer_self.ref, name="binary", attr=Serializable.binary)
        aggressive_set_attr(obj=outer_self.ref, name="to_hex", attr=Serializable.to_hex)
        aggressive_set_attr(obj=outer_self.ref, name="hex", attr=Serializable.hex)
