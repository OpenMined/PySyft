# stdlib
from typing import Any
from typing import Callable as CallableT
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.message import Message

# syft relative
from ..ast.callable import Callable
from ..core.common.serde.serializable import Serializable
from ..core.common.serde.serialize import _serialize
from ..core.node.common.action.run_class_method_action import RunClassMethodAction
from ..core.node.common.action.save_object_action import SaveObjectAction
from ..core.pointer.pointer import Pointer
from ..util import aggressive_set_attr


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
        def get_run_class_method(
            attr_path_and_name: str,
        ) -> object:  # TODO: tighten to return Callable
            """It might seem hugely un-necessary to have these methods nested in this way.
            However, it has to do with ensuring that the scope of attr_path_and_name is local
            and not global. If we do not put a get_run_class_method around run_class_method then
            each run_class_method will end up referencing the same attr_path_and_name variable
            and all methods will actually end up calling the same method. However, if we return
            the function object itself then it includes the current attr_path_and_name as an internal
            variable and when we call get_run_class_method multiple times it returns genuinely
            different methods each time with a different internal attr_path_and_name variable."""

            def run_class_method(
                __self: Any,
                *args: Tuple[Any, ...],
                **kwargs: Any,
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
                        address=__self.client.address,
                    )
                    __self.client.send_immediate_msg_without_reply(msg=cmd)

                return result

            return run_class_method

        attrs = {}
        _props: List[str] = []
        for attr_name, attr in self.attrs.items():
            attr_path_and_name = getattr(attr, "path_and_name", None)

            # if the Method.is_property == True
            # we need to add this attribute name into the _props list
            is_property = getattr(attr, "is_property", False)
            if is_property:
                _props.append(attr_name)
            # QUESTION: Could path_and_name be None?
            # It seems as though attrs can contain
            # Union[Callable, CallableT]
            # where Callable is ast.callable.Callable
            # where CallableT is typing.Callable == any function, method, lambda
            # so we have to check for path_and_name
            if attr_path_and_name is not None:
                attrs[attr_name] = get_run_class_method(attr_path_and_name)

        def getattribute(__self: Any, name: str) -> Any:
            # we need to override the __getattribute__ of our Pointer class
            # so that if you ever access a property on a Pointer it will not just
            # get the wrapped run_class_method but also execute it immediately
            # object.__getattribute__ is the way we prevent infinite recursion
            attr = object.__getattribute__(__self, name)
            props = object.__getattribute__(__self, "_props")

            # if the attr key name is in the _props list from above then we know
            # we should execute it immediately and return the result
            if name in props:
                return attr()

            return attr

        klass_pointer = type(self.pointer_name, (Pointer,), attrs)
        setattr(klass_pointer, "path_and_name", self.path_and_name)
        setattr(klass_pointer, "_props", _props)
        setattr(klass_pointer, "__getattribute__", getattribute)
        setattr(self, self.pointer_name, klass_pointer)

    def create_send_method(outer_self: Any) -> None:
        def send(self: Any, client: Any, searchable: bool = False) -> Pointer:
            # Step 1: create pointer which will point to result
            ptr = getattr(outer_self, outer_self.pointer_name)(
                client=client,
                id_at_location=self.id,
                tags=self.tags if hasattr(self, "tags") else list(),
                description=self.description if hasattr(self, "description") else "",
            )

            # Step 2: create message which contains object to send
            obj_msg = SaveObjectAction(
                obj_id=ptr.id_at_location,
                obj=self,
                address=client.address,
                anyone_can_search_for_this=searchable,
            )

            # Step 3: send message
            client.send_immediate_msg_without_reply(msg=obj_msg)

            # STep 4: return pointer
            return ptr

        def send_to(self: Any, client: Any, searchable: bool = False) -> Pointer:
            # alias method to send method.
            return send(self=self, client=client, searchable=searchable)

        # using curse because Numpy tries to lock down custom attributes
        aggressive_set_attr(obj=outer_self.ref, name="send", attr=send)
        aggressive_set_attr(obj=outer_self.ref, name="send_to", attr=send_to)

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
