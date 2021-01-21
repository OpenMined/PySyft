# stdlib
import inspect
from types import ModuleType
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

# third party
from google.protobuf.message import Message

# syft relative
from .. import ast
from .. import lib
from ..ast.callable import Callable
from ..core.common.serde.serializable import Serializable
from ..core.common.serde.serialize import _serialize
from ..core.common.uid import UID
from ..core.node.common.action.get_or_set_property_action import GetOrSetPropertyAction
from ..core.node.common.action.get_or_set_property_action import PropertyActions
from ..core.node.common.action.run_class_method_action import RunClassMethodAction
from ..core.node.common.action.save_object_action import SaveObjectAction
from ..core.pointer.pointer import Pointer
from ..logger import traceback_and_raise
from ..util import aggressive_set_attr


def get_run_class_method(attr_path_and_name: str) -> CallableT:
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
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass
        return_type_name = __self.client.lib_ast.query(
            attr_path_and_name
        ).return_type_name
        resolved_pointer_type = __self.client.lib_ast.query(return_type_name)
        result = resolved_pointer_type.pointer_type(client=__self.client)

        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)
        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args, kwargs=downcast_kwargs, client=__self.client
            )

            cmd = RunClassMethodAction(
                path=attr_path_and_name,
                _self=__self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                id_at_location=result_id_at_location,
                address=__self.client.address,
            )
            __self.client.send_immediate_msg_without_reply(msg=cmd)

        return result

    return run_class_method


def generate_class_property_function(
    attr_path_and_name: str, action: PropertyActions
) -> CallableT:
    def class_property_function(__self: Any, *args: Any, **kwargs: Any) -> CallableT:
        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass
        return_type_name = __self.client.lib_ast.query(
            attr_path_and_name
        ).return_type_name
        resolved_pointer_type = __self.client.lib_ast.query(return_type_name)
        result = resolved_pointer_type.pointer_type(client=__self.client)
        # QUESTION can the id_at_location be None?
        result_id_at_location = getattr(result, "id_at_location", None)
        if result_id_at_location is not None:
            # first downcast anything primitive which is not already PyPrimitive
            (
                downcast_args,
                downcast_kwargs,
            ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

            # then we convert anything which isnt a pointer into a pointer
            pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
                args=downcast_args, kwargs=downcast_kwargs, client=__self.client
            )

            cmd = GetOrSetPropertyAction(
                path=attr_path_and_name,
                id_at_location=result_id_at_location,
                address=__self.client.address,
                _self=__self,
                args=pointer_args,
                kwargs=pointer_kwargs,
                action=action,
            )
            __self.client.send_immediate_msg_without_reply(msg=cmd)

        return result

    return class_property_function


def _get_request_config(self: Any) -> Dict[str, Any]:
    return {
        "request_block": True,
        "timeout_secs": 25,
        "name": f"__len__ request on {self.id_at_location}",
        "delete_obj": False,
    }


def _set_request_config(self: Any, request_config: Dict[str, Any]) -> None:
    setattr(self, "get_request_config", lambda: request_config)


def wrap_iterator(attrs: Dict[str, Union[str, CallableT, property]]) -> None:
    def wrap_iter(iter_func: CallableT) -> CallableT:
        def __iter__(self: Any) -> Iterable:
            # syft absolute
            from syft.lib.python.iterator import Iterator

            if not hasattr(self, "__len__"):
                traceback_and_raise(
                    ValueError(
                        "Can't build a remote iterator on an object with no __len__."
                    )
                )

            try:
                data_len = len(self)
            except Exception:
                traceback_and_raise(
                    ValueError("Request to access data length not granted.")
                )

            return Iterator(_ref=iter_func(self), max_len=data_len)

        return __iter__

    attr_name = "__iter__"
    iter_target = attrs[attr_name]
    if not callable(iter_target):
        traceback_and_raise(AttributeError("Can't wrap a non callable iter attribute"))
    else:
        iter_func: CallableT = iter_target
    attrs[attr_name] = wrap_iter(iter_func)


def wrap_len(attrs: Dict[str, Union[str, CallableT, property]]) -> None:
    def wrap_len(len_func: CallableT) -> CallableT:
        def __len__(self: Any) -> int:
            data_len_ptr = len_func(self)
            try:
                data_len = data_len_ptr.get(**self.get_request_config())
                return data_len
            except Exception:
                traceback_and_raise(
                    ValueError("Request to access data length not granted.")
                )

        return __len__

    attr_name = "__len__"
    len_target = attrs[attr_name]

    if not callable(len_target):
        traceback_and_raise(
            AttributeError("Can't wrap a non callable __len__ attribute")
        )
    else:
        len_func: CallableT = len_target

    attrs["len"] = len_func
    attrs[attr_name] = wrap_len(len_func)


class Class(Callable):
    def __init__(
        self,
        path_and_name: str,
        object_ref: Union[Callable, CallableT],
        return_type_name: Optional[str],
        client: Optional[Any],
    ):
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
        )
        if self.path_and_name is not None:
            self.pointer_name = self.path_and_name.split(".")[-1] + "Pointer"

    @property
    def pointer_type(self) -> Union[Callable, CallableT]:
        return getattr(self, self.pointer_name)

    def create_pointer_class(self) -> None:
        attrs: Dict[str, Union[str, CallableT, property]] = {}
        for attr_name, attr in self.attrs.items():
            attr_path_and_name = getattr(attr, "path_and_name", None)

            # attr_path_and_name None
            if isinstance(attr, ast.callable.Callable):
                attrs[attr_name] = get_run_class_method(attr_path_and_name)
            elif isinstance(attr, ast.property.Property):
                prop = property(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.GET
                    )
                )

                prop = prop.setter(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.SET
                    )
                )
                prop = prop.deleter(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.DEL
                    )
                )
                attrs[attr_name] = prop

            if attr_name == "__len__":
                wrap_len(attrs)

            if getattr(attr, "return_type_name", None) == "syft.lib.python.Iterator":
                wrap_iterator(attrs)

        attrs["get_request_config"] = _get_request_config
        attrs["set_request_config"] = _set_request_config

        fqn = "Pointer"

        if self.path_and_name is not None:
            fqn = self.path_and_name + fqn

        new_class_name = f"syft.proxy.{fqn}"
        parts = new_class_name.split(".")
        name = parts.pop(-1)
        attrs["__name__"] = name
        attrs["__module__"] = ".".join(parts)

        klass_pointer = type(self.pointer_name, (Pointer,), attrs)
        setattr(klass_pointer, "path_and_name", self.path_and_name)
        setattr(self, self.pointer_name, klass_pointer)

    def create_send_method(outer_self: Any) -> None:
        def send(
            self: Any,
            client: Any,
            searchable: bool = False,
            description: str = "",
            tags: List[str] = [],
        ) -> Pointer:
            # if self is proto, change self to it's wrapper object
            which_obj = self
            if "ProtobufWrapper" in self.serializable_wrapper_type.__name__:
                # which_obj should be of the same type as what self._data_proto2object returns
                which_obj = self.serializable_wrapper_type(value=self)

            if "CTypeWrapper" in self.serializable_wrapper_type.__name__:
                # which_obj should be of the same type as what self._data_proto2object returns
                which_obj = self.serializable_wrapper_type(value=self)

            id_ = getattr(self, "id", None)
            if id_ is None:
                id_ = UID()
                which_obj.id = id_
            which_obj.tags = tags
            which_obj.description = description

            id_at_location = UID()

            # Step 1: create pointer which will point to result
            ptr = getattr(outer_self, outer_self.pointer_name)(
                client=client,
                id_at_location=id_at_location,
                tags=tags,
                description=description,
            )

            if searchable:
                ptr.gc_enabled = False

            # Step 2: create message which contains object to send
            obj_msg = SaveObjectAction(
                id_at_location=ptr.id_at_location,
                obj=which_obj,
                address=client.address,
                anyone_can_search_for_this=searchable,
            )

            # Step 3: send message
            client.send_immediate_msg_without_reply(msg=obj_msg)

            # Step 4: return pointer
            return ptr

        aggressive_set_attr(obj=outer_self.object_ref, name="send", attr=send)

    def create_storable_object_attr_convenience_methods(outer_self: Any) -> None:
        def tag(self: Any, *tags: Tuple[Any, ...]) -> object:
            self.tags = list(tags)
            return self

        def describe(self: Any, description: str) -> object:
            self.description = description
            # QUESTION: Is this supposed to return self?
            # WHY? Chaining?
            return self

        aggressive_set_attr(obj=outer_self.object_ref, name="tag", attr=tag)
        aggressive_set_attr(obj=outer_self.object_ref, name="describe", attr=describe)

    def create_serialization_methods(outer_self) -> None:
        def serialize(  # type: ignore
            self,
            to_proto: bool = True,
            to_bytes: bool = False,
        ) -> Union[str, bytes, Message]:
            return _serialize(
                obj=self,
                to_proto=to_proto,
                to_bytes=to_bytes,
            )

        serialize_attr = "serialize"
        if not hasattr(outer_self.object_ref, serialize_attr):
            aggressive_set_attr(
                obj=outer_self.object_ref, name=serialize_attr, attr=serialize
            )
        else:
            serialize_attr = "sy_serialize"
            aggressive_set_attr(
                obj=outer_self.object_ref, name=serialize_attr, attr=serialize
            )

        aggressive_set_attr(
            obj=outer_self.object_ref, name="to_proto", attr=Serializable.to_proto
        )
        aggressive_set_attr(
            obj=outer_self.object_ref, name="proto", attr=Serializable.proto
        )
        to_bytes_attr = "to_bytes"
        # int has a to_bytes already, so we can use _to_bytes internally
        if hasattr(outer_self.object_ref, to_bytes_attr):
            to_bytes_attr = "_to_bytes"
        aggressive_set_attr(
            obj=outer_self.object_ref, name=to_bytes_attr, attr=Serializable.to_bytes
        )

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:

        if index >= len(path) or path[index] in self.attrs:
            return

        _path: List[str] = path.split(".") if isinstance(path, str) else path

        # stdlib
        from enum import Enum
        from enum import EnumMeta

        attr_ref = getattr(self.object_ref, _path[index])

        class_is_enum = isinstance(self.object_ref, EnumMeta)

        if (
            inspect.isfunction(attr_ref)
            or inspect.isbuiltin(attr_ref)
            or inspect.ismethod(attr_ref)
            or inspect.ismethoddescriptor(attr_ref)
        ):
            super().add_path(_path, index, return_type_name)
        if isinstance(attr_ref, Enum) and class_is_enum:
            enum_attribute = ast.enum.EnumAttribute(
                path_and_name=".".join(_path[: index + 1]),
                return_type_name=return_type_name,
                client=self.client,
                parent=self,
            )
            setattr(self, _path[index], enum_attribute)
            self.attrs[_path[index]] = enum_attribute

        elif inspect.isdatadescriptor(attr_ref) or inspect.isgetsetdescriptor(attr_ref):
            self.attrs[_path[index]] = ast.property.Property(
                path_and_name=".".join(_path[: index + 1]),
                object_ref=attr_ref,
                return_type_name=return_type_name,
                client=self.client,
            )
        elif not callable(attr_ref):
            static_attribute = ast.static_attr.StaticAttribute(
                path_and_name=".".join(_path[: index + 1]),
                return_type_name=return_type_name,
                client=self.client,
                parent=self,
            )
            setattr(self, _path[index], static_attribute)
            self.attrs[_path[index]] = static_attribute

    def __getattribute__(self, item: str) -> Any:
        try:
            target_object = super().__getattribute__(item)

            if isinstance(target_object, ast.static_attr.StaticAttribute):
                return target_object.get_remote_value()

            if isinstance(target_object, ast.enum.EnumAttribute):
                target_object_ptr = target_object.get_remote_enum_attribute()
                target_object_ptr.is_enum = True
                return target_object_ptr

            return target_object
        except Exception as e:
            raise e

    def __getattr__(self, item: str) -> Any:
        attrs = super().__getattribute__("attrs")
        return attrs[item] if item in attrs else None

    def __setattr__(self, key: str, value: Any) -> None:
        if hasattr(super(), "attrs"):
            attrs = super().__getattribute__("attrs")
            if key in attrs:
                target_object = self.attrs[key]
                if isinstance(target_object, ast.static_attr.StaticAttribute):
                    return target_object.set_remote_value(value)

        return super().__setattr__(key, value)


def pointerize_args_and_kwargs(
    args: Union[List[Any], Tuple[Any, ...]], kwargs: Dict[Any, Any], client: Any
) -> Tuple[List[Any], Dict[Any, Any]]:
    # When we try to send params to a remote function they need to be pointers so
    # that they can be serialized and fetched from the remote store on arrival
    # this ensures that any args which are passed in from the user side are first
    # converted to pointers and sent then the pointer values are used for the
    # method invocation
    pointer_args = []
    pointer_kwargs = {}
    for arg in args:
        # check if its already a pointer
        if not isinstance(arg, Pointer):
            arg_ptr = arg.send(client)
            pointer_args.append(arg_ptr)
        else:
            pointer_args.append(arg)

    for k, arg in kwargs.items():
        # check if its already a pointer
        if not isinstance(arg, Pointer):
            arg_ptr = arg.send(client)
            pointer_kwargs[k] = arg_ptr
        else:
            pointer_kwargs[k] = arg

    return (pointer_args, pointer_kwargs)
