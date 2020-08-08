from forbiddenfruit import curse

from .. import ast
from ..core import pointer as ptr
from ..core.node.common.action.run_class_method_action import RunClassMethodAction
from ..core.node.common.action.save_object_action import SaveObjectAction
from ..core.common.serde.serializable import Serializable
from ..core.common.serde.serialize import _serialize
from google.protobuf.message import Message
from typing import Union
from ..util import aggressive_set_attr


class Class(ast.callable.Callable):

    ""

    def __init__(self, name, path_and_name, ref, return_type_name):
        super().__init__(name, path_and_name, ref, return_type_name=return_type_name)
        self.pointer_name = self.path_and_name + "Pointer"

    def __repr__(self):
        return f"<Class:{self.name}>"

    @property
    def pointer_type(self):
        return getattr(self, self.pointer_name)

    def create_pointer_class(self):
        def run_class_method(attr, attr_path_and_name, __self, args, kwargs):
            # TODO: lookup actual return type instead of just guessing that it's identical
            result = getattr(self, self.pointer_name)(location=__self.location)

            cmd = RunClassMethodAction(
                path=attr_path_and_name,
                _self=__self,
                args=args,
                kwargs=kwargs,
                id_at_location=result.id_at_location,
                address=__self.location.address,
            )
            __self.location.send_immediate_msg_without_reply(msg=cmd)

            return result

        attrs = {}
        for attr_name, attr in self.attrs.items():
            attrs[attr_name] = lambda _self, *args, **kwargs: run_class_method(
                attr, attr.path_and_name, _self, args, kwargs
            )

        klass_pointer = type(self.pointer_name, (ptr.pointer.Pointer,), attrs)

        setattr(self, self.pointer_name, klass_pointer)

    def create_send_method(outer_self):
        def send(self, location):
            # Step 1: create pointer which will point to result
            ptr = getattr(outer_self, outer_self.pointer_name)(location=location)

            # Step 2: create old_message which contains object to send
            obj_msg = SaveObjectAction(
                obj_id=ptr.id_at_location, obj=self, address=location.address
            )

            # Step 3: send old_message
            location.send_immediate_msg_without_reply(msg=obj_msg)

            # STep 4: return pointer
            return ptr

        # using curse because Numpy tries to lock down custom attributes
        aggressive_set_attr(obj=outer_self.ref, name="send", attr=send)

    def create_serialization_methods(outer_self):
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

        curse(outer_self.ref, "serialize", serialize)
        curse(outer_self.ref, "to_proto", Serializable.to_proto)
        curse(outer_self.ref, "proto", Serializable.proto)
        curse(outer_self.ref, "to_json", Serializable.to_json)
        curse(outer_self.ref, "json", Serializable.json)
        curse(outer_self.ref, "to_binary", Serializable.to_binary)
        curse(outer_self.ref, "binary", Serializable.binary)
        curse(outer_self.ref, "to_hex", Serializable.to_hex)
        curse(outer_self.ref, "hex", Serializable.hex)
