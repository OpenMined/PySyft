# stdlib
import sys
from typing import Any
from typing import Callable as CallableT
from typing import Dict
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
            print("what is our attr_path and name", attr_path_and_name)

            def run_class_method(
                __self: Any,
                *args: Tuple[Any, ...],
                **kwargs: Any,
            ) -> object:
                # result = self.pointer_type(client=__self.client)

                return_type_name = __self.client.lib_ast(
                    attr_path_and_name, return_callable=True
                ).return_type_name
                resolved_pointer_type = __self.client.lib_ast(
                    return_type_name, return_callable=True
                )
                result = resolved_pointer_type.pointer_type(client=__self.client)

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
                print("call property", name, attr)
                return attr()

            return attr

        # here we can ensure that the fully qualified name of the Pointer klass is
        # consistent between versions of python and matches our other klasses in
        # generic.py like subclassed or ShadowWrapper constructors
        # this will result in: syft.proxy.{original_fully_qualified_name}Pointer
        fqn = "Pointer"
        # this should always be a str
        if self.path_and_name is not None:
            # prepend
            fqn = self.path_and_name + fqn
        new_class_name = f"syft.proxy.{fqn}"
        parts = new_class_name.split(".")
        name = parts.pop(-1)
        attrs["__name__"] = name
        attrs["__module__"] = ".".join(parts)

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


# The ClassFactory replaces the previous usage of initalizing a klass.Class during ast.
# The difference is that firstly, while building the ast a subclass of Class will be
# created however the initializer will be swapped out so that if a person subclasses
# the externally facing module path class name like so:
#
# class MyTensor(torch.Tensor):
#     pass
#
# The result will be that the creation of the class won't raise an Exception since
# torch.Tensor now points to klass.Class which expects name, path_and_name etc and worse
# its an object with __call__ not a class, so subclassing it will be missing all the
# a bunch of initialized data needed to work. The second step is that when manually
# defining the new class MyTensor(torch.Tensor) as above, the hijack code path will
# activate and the sys.modules["torch"].original_Tensor will be returned thus providing
# something to initialize with the expected __init__ params and returning a sane obj.
# This is WIP and the next step is to modify the __name__ and __module__ and mro() to
# give the expected experience of subclassing and finally figure out how to what extra
# hurdles will be required to get this working.


def ClassFactory(
    name: Optional[str],
    path_and_name: Optional[str],
    ref: Union[Callable, CallableT],
    return_type_name: Optional[str],
) -> Class:
    attrs: Dict[Any, Any] = {}

    def new(
        cls: type,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> None:
        if len(args) == 0:
            new_cls = super(Class, cls).__new__(cls)  # type: ignore
            new_cls._hijack = False
        else:
            if path_and_name is not None:
                parts = path_and_name.split(".")
                end = parts.pop(-1)
                end = f"original_{end}"
                sub: Any = sys.modules
                for part in parts:
                    if type(sub) is dict:
                        if part in sub:
                            sub = sub[part]
                    else:
                        if hasattr(sub, part):
                            sub = getattr(sub, part, None)

                org_class = getattr(sub, end, None)
                return org_class

        return new_cls

    attrs["__new__"] = new

    def default_args_init(
        _self: Any,
        *args: Tuple[Any, ...],
        **kwargs: Any,
    ) -> None:
        hijack = getattr(_self, "_hijack", None)
        if hijack is False:
            Class.__init__(
                _self,
                name=name,
                path_and_name=path_and_name,
                ref=ref,
                return_type_name=return_type_name,
            )

    # default_args_init.__text_signature__ = "($self, *args, **kwargs)"  # type: ignore
    attrs["__init__"] = default_args_init
    SubclassableClass = type("SubclassableClass", (Class,), attrs)
    return SubclassableClass()
