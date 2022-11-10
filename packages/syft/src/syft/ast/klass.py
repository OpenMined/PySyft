"""This module contains `Class` attribute,an AST node representing a class."""

# stdlib
from enum import Enum
from enum import EnumMeta
import inspect
import sys
from types import ModuleType
from typing import Any
from typing import Callable as CallableT
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

# relative
from .. import ast
from .. import lib
from ..core.common.group import VERIFYALL
from ..core.common.uid import UID
from ..core.node.common.action.action_sequence import ActionSequence
from ..core.node.common.action.get_or_set_property_action import GetOrSetPropertyAction
from ..core.node.common.action.get_or_set_property_action import PropertyActions
from ..core.node.common.action.run_class_method_action import RunClassMethodAction
from ..core.node.common.action.run_class_method_smpc_action import (
    RunClassMethodSMPCAction,
)
from ..core.node.common.action.save_object_action import SaveObjectAction
from ..core.node.common.node_service.resolve_pointer_type.resolve_pointer_type_messages import (
    ResolvePointerTypeMessage,
)
from ..core.node.common.util import check_send_to_blob_storage
from ..core.node.common.util import upload_to_s3_using_presigned
from ..core.pointer.pointer import Pointer
from ..core.store.storeable_object import StorableObject
from ..logger import traceback_and_raise
from ..logger import warning
from ..util import aggressive_set_attr
from ..util import inherit_tags
from .callable import Callable


def _resolve_pointer_type(self: Pointer) -> Pointer:
    """Resolve pointer of the object.

    Creates a request on a pointer to validate and regenerate the current pointer type. This method
    is useful when dealing with AnyPointer or Union<types>Pointers, to retrieve the real pointer.

    The existing pointer will be deleted and a new one will be generated. The remote data won't
    be touched.

    Args:
        self: The pointer which will be validated.

    Returns:
        The new pointer, validated from the remote object.
    """
    # id_at_location has to be preserved
    id_at_location = getattr(self, "id_at_location", None)

    if id_at_location is None:
        traceback_and_raise(
            ValueError("Can't resolve a pointer that has no underlying object.")
        )

    cmd = ResolvePointerTypeMessage(
        id_at_location=id_at_location,
        address=self.client.address,
        reply_to=self.client.address,
    )

    # the path to the underlying type. It has to live in the AST
    real_type_path = self.client.send_immediate_msg_with_reply(msg=cmd).type_path
    new_pointer = self.client.lib_ast.query(real_type_path).pointer_type(
        client=self.client, id_at_location=id_at_location
    )

    # we disable the garbage collection message and then we delete the existing message.
    self.gc_enabled = False
    del self

    return new_pointer


def get_run_class_method(attr_path_and_name: str, SMPC: bool = False) -> CallableT:
    """Create a function for class method in `attr_path_and_name` for remote execution.

    Args:
        attr_path_and_name: The path of the class method.

    Returns:
        Function for the class method.

    Note:
        It might seem hugely un-necessary to have these methods nested in this way.
        However, it has to do with ensuring that the scope of `attr_path_and_name` is local
        and not global.

        If we do not put a `get_run_class_method` around `run_class_method` then
        each `run_class_method` will end up referencing the same `attr_path_and_name` variable
        and all methods will actually end up calling the same method.

        If, instead, we return the function object itself then it includes
        the current `attr_path_and_name` as an internal variable and when we call `get_run_class_method`
        multiple times it returns genuinely different methods each time with a different
        internal `attr_path_and_name` variable.
    """
    # relative
    from ..core.node.common.action import smpc_action_functions

    def run_class_smpc_method(
        __self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        """Run remote class method on a SharePointer and get pointer to returned object.

        Args:
            *args: Args list of class method.
            **kwargs: Keyword args of class method.

        Returns:
            Pointer to object returned by class method.
        """
        seed_id_locations = kwargs.pop("seed_id_locations", None)
        if seed_id_locations is None:
            raise ValueError(
                "There should be a `seed_id_locations` kwargs when doing an operation for MPCTensor"
            )

        op = attr_path_and_name.split(".")[-1]
        id_at_location = smpc_action_functions.get_id_at_location_from_op(
            seed_id_locations, op
        )

        # we want to get the return type which matches the attr_path_and_name
        # so we ask lib_ast for the return type name that matches out
        # attr_path_and_name and then use that to get the actual pointer klass
        # then set the result to that pointer klass
        return_type_name = __self.client.lib_ast.query(
            attr_path_and_name
        ).return_type_name
        resolved_pointer_type = __self.client.lib_ast.query(return_type_name)
        result = resolved_pointer_type.pointer_type(client=__self.client)
        result.id_at_location = id_at_location

        # first downcast anything primitive which is not already PyPrimitive
        (
            downcast_args,
            downcast_kwargs,
        ) = lib.python.util.downcast_args_and_kwargs(args=args, kwargs=kwargs)

        # then we convert anything which isnt a pointer into a pointer
        pointer_args, pointer_kwargs = pointerize_args_and_kwargs(
            args=downcast_args,
            kwargs=downcast_kwargs,
            client=__self.client,
            gc_enabled=False,
        )

        cmd = RunClassMethodSMPCAction(
            path=attr_path_and_name,
            _self=__self,
            args=pointer_args,
            kwargs=pointer_kwargs,
            id_at_location=result.id_at_location,
            seed_id_locations=seed_id_locations,
            address=__self.client.address,
        )
        __self.client.send_immediate_msg_without_reply(msg=cmd)

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=__self,
            args=args,
            kwargs=kwargs,
        )

        return result

    def run_class_method(
        __self: Any,
        *args: Any,
        **kwargs: Any,
    ) -> object:
        """Run remote class method and get pointer to returned object.

        Args:
            *args: Args list of class method.
            **kwargs: Keyword args of class method.

        Returns:
            Pointer to object returned by class method.
        """

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
                args=downcast_args,
                kwargs=downcast_kwargs,
                client=__self.client,
                gc_enabled=False,
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

        inherit_tags(
            attr_path_and_name=attr_path_and_name,
            result=result,
            self_obj=__self,
            args=args,
            kwargs=kwargs,
        )

        return result

    method_name = attr_path_and_name.rsplit(".", 1)[-1]
    if SMPC or (
        "ShareTensor" in attr_path_and_name
        and method_name in smpc_action_functions.ACTION_FUNCTIONS
    ):
        return run_class_smpc_method

    return run_class_method


def generate_class_property_function(
    attr_path_and_name: str, action: PropertyActions, map_to_dyn: bool
) -> CallableT:
    """Returns a function that handles action on property.

    Args:
        attr_path_and_name: The path of the property in AST.
        action: action to perform on property (GET | SET | DEL).

    Returns:
        Function to handle action on property.
    """

    def class_property_function(__self: Any, *args: Any, **kwargs: Any) -> object:
        """Handles remote action on property and returns pointer.

        Args:
            *args: Argument list.
            **kwargs: Keyword arguments.

        Returns:
            Pointer to the object returned.
        """
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
                map_to_dyn=map_to_dyn,
            )
            __self.client.send_immediate_msg_without_reply(msg=cmd)

        if action == PropertyActions.GET:
            inherit_tags(
                attr_path_and_name=attr_path_and_name,
                result=result,
                self_obj=__self,
                args=args,
                kwargs=kwargs,
            )
        return result

    return class_property_function


def _get_request_config(self: Any) -> Dict[str, Any]:
    """Get config for request.

    Args:
        self: object.

    Returns:
        Config for request.
    """
    return {
        "request_block": True,
        "timeout_secs": 25,
        "delete_obj": False,
    }


def _set_request_config(self: Any, request_config: Dict[str, Any]) -> None:
    """Set config for request.

    Args:
        self: object.
        request_config: new config.
    """
    self.get_request_config = lambda: request_config


def wrap_iterator(attrs: Dict[str, Union[str, CallableT, property]]) -> None:
    """Add syft Iterator to `attrs['__iter__']`.

    Args:
        attrs: Dict of `Attribute`s of node.

    Raises:
        AttributeError: Base `__iter__` is not callable.
    """

    def wrap_iter(iter_func: CallableT) -> CallableT:
        """Create syft iterator for `iter_func`.

        Args:
            iter_func: Base Iterator.

        Returns:
            Wrapped Iterator.
        """

        def __iter__(self: Any) -> Iterable:
            """Create Syft Iterator for `iter_func`.

            Args:
                self: object to add iterator to.

            Raises:
                ValueError: Falied ot access __len__.

            Returns:
                Iterable: syft Iterator.
            """
            # relative
            from ..lib.python.iterator import Iterator

            if not hasattr(self, "__len__"):
                traceback_and_raise(
                    ValueError(
                        "Can't build a remote iterator on an object with no __len__."
                    )
                )

            try:
                data_len = self.__len__()
            except Exception:
                traceback_and_raise(
                    ValueError("Request to access data length rejected.")
                )

            return Iterator(_ref=iter_func(self), max_len=data_len)

        return __iter__

    attr_name = "__iter__"
    iter_target = attrs[attr_name]

    # skip if __iter__ has already been wrapped
    qual_name = getattr(iter_target, "__qualname__", None)
    if qual_name and "wrap_iter" in qual_name:
        return

    if not callable(iter_target):
        traceback_and_raise(AttributeError("Can't wrap a non callable iter attribute"))
    else:
        iter_func: CallableT = iter_target
    attrs[attr_name] = wrap_iter(iter_func)


def wrap_len(attrs: Dict[str, Union[str, CallableT, property]]) -> None:
    """Add method to access pointer len to `attr[__len__]`.

    Args:
        attrs: Dict of `Attribute`s of node.

    Raises:
        AttributeError: Base `__len__` is not callable.
    """

    def wrap_len(len_func: CallableT) -> CallableT:
        """Add wrapper function for `len_func`.

        Args:
            len_func: Base len function.

        Returns:
            Wrapped len function.
        """

        def __len__(self: Any) -> int:
            """Access len of pointer obj.

            Args:
                self: object to add iterator to.

            Returns:
                int: length of object.

            Raises:
                ValueError: Request to access data length rejected.
            """
            data_len_ptr = len_func(self)
            try:
                print(self.get_request_config())
                data_len = data_len_ptr.get(**self.get_request_config())

                if data_len is None:
                    raise Exception

                return data_len
            except Exception:
                traceback_and_raise(
                    ValueError("Request to access data length rejected.")
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


def attach_tags(obj: object, tags: List[str]) -> None:
    """Add tags to the object.

    Args:
        obj: Object to add tags to.
        tags: List of tags.

    Raises:
        AttributeError: Cannot add tags to object.
    """
    try:
        obj.tags = sorted(set(tags), key=tags.index)  # type: ignore
    except AttributeError:
        warning(f"can't attach new attribute `tags` to {type(obj)} object.")


def attach_description(obj: object, description: str) -> None:
    """Add description to the object.

    Args:
        obj: Object to add description to.
        description: Description.

    Raises:
        AttributeError: Cannot add description to object.
    """
    try:
        obj.description = description  # type: ignore
    except AttributeError:
        warning(f"can't attach new attribute `description` to {type(obj)} object.")


class Class(Callable):
    """A Class attribute represents a class."""

    def __init__(
        self,
        path_and_name: str,
        parent: ast.attribute.Attribute,
        object_ref: Union[Callable, CallableT],
        return_type_name: Optional[str],
        client: Optional[Any],
    ) -> None:
        """Base constructor for Class Attribute.

        Args:
            path_and_name: The path for the current node, e.g. `syft.lib.python.List`.
            parent: The parent node is needed when solving `EnumAttributes`.
            object_ref: The actual python object for which the computation is being made.
            return_type_name: The return type name of given action as a string with its full path.
            client: The client for which all computation is being executed.
        """
        super().__init__(
            path_and_name=path_and_name,
            object_ref=object_ref,
            return_type_name=return_type_name,
            client=client,
            parent=parent,
        )
        if self.path_and_name is not None:
            self.pointer_name = self.path_and_name.split(".")[-1] + "Pointer"

    @property
    def pointer_type(self) -> Union[Callable, CallableT]:
        """Get pointer type of Class Attribute.

        Returns:
            `pointer_type` of the object.
        """
        return getattr(self, self.pointer_name)

    def create_pointer_class(self) -> None:
        """Create pointer type for object."""
        attrs: Dict[str, Union[str, CallableT, property]] = {}
        for attr_name, attr in self.attrs.items():
            attr_path_and_name: Optional[str] = getattr(attr, "path_and_name", None)

            if attr_path_and_name is None:
                raise Exception(f"Missing path_and_name in {self.attrs}")

            # attr_path_and_name None
            if isinstance(attr, ast.callable.Callable):
                attrs[attr_name] = get_run_class_method(attr_path_and_name)
            elif isinstance(attr, ast.property.Property):
                prop = property(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.GET, map_to_dyn=False
                    )
                )

                prop = prop.setter(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.SET, map_to_dyn=False
                    )
                )
                prop = prop.deleter(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.DEL, map_to_dyn=False
                    )
                )
                attrs[attr_name] = prop
            elif isinstance(attr, ast.dynamic_object.DynamicObject):
                prop = property(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.GET, map_to_dyn=True
                    )
                )

                prop = prop.setter(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.SET, map_to_dyn=True
                    )
                )
                prop = prop.deleter(
                    generate_class_property_function(
                        attr_path_and_name, PropertyActions.DEL, map_to_dyn=True
                    )
                )
                attrs[attr_name] = prop
            if attr_name == "__len__":
                wrap_len(attrs)

            if getattr(attr, "return_type_name", None) == "syft.lib.python.Iterator":
                wrap_iterator(attrs)

        attrs["get_request_config"] = _get_request_config
        attrs["set_request_config"] = _set_request_config
        attrs["resolve_pointer_type"] = _resolve_pointer_type

        fqn = "Pointer"

        if self.path_and_name is not None:
            fqn = self.path_and_name + fqn

        new_class_name = f"syft.proxy.{fqn}"
        parts = new_class_name.split(".")
        name = parts.pop(-1)
        attrs["__name__"] = name
        attrs["__module__"] = ".".join(parts)

        # if the object already has a pointer class specified, use that instead of creating
        # an empty subclass of Pointer
        if hasattr(self.object_ref, "PointerClassOverride"):

            klass_pointer = getattr(self.object_ref, "PointerClassOverride")
            for key, val in attrs.items():

                # only override functioanlity of AST attributes if they
                # don't already exist on the PointerClassOverride class
                # (the opposite of inheritance)
                if not hasattr(klass_pointer, key):
                    setattr(klass_pointer, key, val)
                else:
                    # TODO: cache attribute in backup_ location so that we can use them if we want
                    pass

        # no specific pointer class found, let's make an empty subclass of Pointer instead
        else:
            klass_pointer = type(self.pointer_name, (Pointer,), attrs)

        setattr(klass_pointer, "path_and_name", self.path_and_name)
        setattr(self, self.pointer_name, klass_pointer)

        module_type = type(sys)

        # syft absolute
        import syft

        parent = syft
        for part in parts[1:]:
            if part not in parent.__dict__:
                parent.__dict__[part] = module_type(name=part)
            parent = parent.__dict__[part]
        parent.__dict__[name] = klass_pointer

    def store_init_args(outer_self: Any) -> None:
        """
        Stores args and kwargs of outer_self init by wrapping the init method.
        """

        def init_wrapper(self: Any, *args: Any, **kwargs: Any) -> None:
            outer_self.object_ref._wrapped_init(self, *args, **kwargs)
            self._init_args = args
            self._init_kwargs = kwargs

        # If _wrapped_init already exists, create_init_method is already called once
        # and does not need to wrap __init__ again.
        if not hasattr(outer_self.object_ref, "_wrapped_init"):
            outer_self.object_ref._wrapped_init = outer_self.object_ref.__init__
            outer_self.object_ref.__init__ = init_wrapper

    def create_send_method(outer_self: Any) -> None:
        """Add `send` method to `outer_self.object_ref`."""

        def send(
            self: Any,
            client: Any,
            pointable: bool = True,
            description: str = "",
            tags: Optional[List[str]] = None,
            searchable: Optional[bool] = None,
            id_at_location_override: Optional[UID] = None,
            chunk_size: Optional[int] = None,
            send_to_blob_storage: bool = True,
            **kwargs: Any,
        ) -> Union[Pointer, Tuple[Pointer, SaveObjectAction]]:

            """Send obj to client and return pointer to the object.

            Args:
                self: Object to be sent.
                client: Client to send object to.
                pointable:
                description: Description for the object to send.
                tags: Tags for the object to send.

            Returns:
                Pointer to sent object.

            Note:
                `searchable` is deprecated please use `pointable` in the future.
            """
            if searchable is not None:
                msg = "`searchable` is deprecated please use `pointable` in future"
                warning(msg, print=True)
                warnings.warn(
                    msg,
                    DeprecationWarning,
                )
                pointable = searchable

            chunk_size = chunk_size if chunk_size is not None else 536870912  # 500 MB

            if not hasattr(self, "id"):
                try:
                    self.id = UID()
                except AttributeError:
                    pass

            # if `tags` is passed in, use it; else, use obj_tags
            obj_tags = getattr(self, "tags", [])
            tags = tags if tags else []
            tags = tags if tags else obj_tags

            # if `description` is passed in, use it; else, use obj_description
            obj_description = getattr(self, "description", "")
            description = description if description else obj_description

            # TODO: Allow Classes to opt out in the AST like Pandas where the properties
            # would break their dict attr usage
            # Issue: https://github.com/OpenMined/PySyft/issues/5322
            if outer_self.pointer_name not in {"DataFramePointer", "SeriesPointer"}:
                attach_tags(self, tags)
                attach_description(self, description)

            if id_at_location_override is not None:
                id_at_location = id_at_location_override
            else:
                id_at_location = UID()

            if hasattr(self, "init_pointer"):
                constructor = self.init_pointer
            else:
                constructor = getattr(outer_self, outer_self.pointer_name)

            # Step 1: create pointer which will point to result
            ptr = constructor(
                client=client,
                id_at_location=id_at_location,
                tags=tags,
                description=description,
            )

            ptr._pointable = pointable

            if pointable:
                ptr.gc_enabled = False
            else:
                ptr.gc_enabled = True

            # Check if the client has blob storage enabled
            # blob storage can only be used if client node has blob storage enabled.
            if not hasattr(client, "settings") or not client.settings.get(
                "use_blob_storage", False
            ):
                sys.stdout.write(
                    "\n**Warning**: Blob Storage is disabled on this client node. Switching to database store.\n"
                )
                send_to_blob_storage = False

            # Check if the obj satisfies the min requirements for it to be stored in blob store
            store_obj_in_blob_store = check_send_to_blob_storage(
                obj=self, use_blob_storage=send_to_blob_storage
            )

            if store_obj_in_blob_store:
                store_data = upload_to_s3_using_presigned(
                    client=client,
                    data=self,
                    chunk_size=chunk_size,
                    asset_name=id_at_location.no_dash,
                )
            else:
                store_data = self

            # Step 6: create message which contains object to send
            storable = StorableObject(
                id=ptr.id_at_location,
                data=store_data,
                tags=tags,
                description=description,
                search_permissions={VERIFYALL: None} if pointable else {},
            )
            obj_msg = SaveObjectAction(obj=storable, address=client.address)

            immediate = kwargs.get("immediate", True)

            if immediate:
                # Step 7: send message
                client.send_immediate_msg_without_reply(msg=obj_msg)

                # Step 8: return pointer
                return ptr
            else:
                return ptr, obj_msg

        aggressive_set_attr(obj=outer_self.object_ref, name="send", attr=send)

    def create_storable_object_attr_convenience_methods(outer_self: Any) -> None:
        """Add methods to set tag and description to `outer_self.object_ref`."""

        def tag(self: Any, *tags: Tuple[Any, ...]) -> object:
            """Add tags to object.

            Args:
                self: object to add tags to.
                *tags: List of tags to add.

            Returns:
                object.
            """
            attach_tags(self, tags)  # type: ignore
            return self

        def describe(self: Any, description: str) -> object:
            """Add description to object.

            Args:
                self: object to add description to.
                description: Description to add.

            Returns:
                object.
            """
            attach_description(self, description)
            return self

        aggressive_set_attr(obj=outer_self.object_ref, name="tag", attr=tag)
        aggressive_set_attr(obj=outer_self.object_ref, name="describe", attr=describe)

    def add_path(
        self,
        path: Union[str, List[str]],
        index: int,
        return_type_name: Optional[str] = None,
        framework_reference: Optional[ModuleType] = None,
        is_static: bool = False,
    ) -> None:
        """The add_path method adds new nodes in AST based on type of current node and type of object to be added.

        Args:
            path: The node path added in AST, e.g. `syft.lib.python.List` or ["syft", "lib", "python", "List].
            index: The associated position in the path for the current node.
            framework_reference: The Python framework in which we can resolve same path to obtain Python object.
            return_type_name: The return type name of the given action as a string with its full path.
            is_static: If the queried object is static, it has to be found on AST itself, not on an existing pointer.
        """
        if index >= len(path) or path[index] in self.attrs:
            return

        _path: List[str] = path.split(".") if isinstance(path, str) else path
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
                parent=self,
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

    def add_dynamic_object(self, path_and_name: str, return_type_name: str) -> None:
        self.attrs[
            path_and_name.rsplit(".", maxsplit=1)[-1]
        ] = ast.dynamic_object.DynamicObject(
            path_and_name=path_and_name,
            return_type_name=return_type_name,
            client=self.client,
            parent=self,
        )

    def __getattribute__(self, item: str) -> Any:
        """Get pointer to attribute.

        Args:
            item: Attribute.

        Returns:
            Pointer to the attribute.
        """
        # self.apply_node_changes()
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
            # TODO: this gets really chatty when doing SMPC mulitplication. Figure out why.
            # critical(
            #     f"{self.path_and_name}__getattribute__[{item}] failed. If you
            #     are trying to access an EnumAttribute or a "
            #     "StaticAttribute, be sure they have been added to the AST. Falling back on"
            #     "__getattr__ to search in self.attrs for the requested field."
            # )
            traceback_and_raise(e)

    def __getattr__(self, item: str) -> Any:
        """Get value of attribute `item` of the object.

        Args:
            item: Attribute.

        Raises:
            KeyError: If attribute `item` is not present.

        Returns:
            Value of the attribute.
        """
        attrs = super().__getattribute__("attrs")
        if item not in attrs:
            if item == "__name__":
                # return the pointer name if __name__ is missing
                return self.pointer_name
            traceback_and_raise(
                KeyError(
                    f"__getattr__ failed, {item} is not present on the "
                    f"object, nor the AST attributes!"
                )
            )
        return attrs[item]

    def __setattr__(self, key: str, value: Any) -> None:
        """Change value of attribute `key` to `value`.

        Args:
            key: name of attribute to change.
            value: value to change attribute `key` to.
        """
        # self.apply_node_changes()

        if hasattr(super(), "attrs"):
            attrs = super().__getattribute__("attrs")
            if key in attrs:
                target_object = self.attrs[key]
                if isinstance(target_object, ast.static_attr.StaticAttribute):
                    return target_object.set_remote_value(value)

        return super().__setattr__(key, value)


# TODO: this should move out of AST into a util somewhere? or osmething related to Pointer
def pointerize_args_and_kwargs(
    args: Union[List[Any], Tuple[Any, ...]],
    kwargs: Dict[Any, Any],
    client: Any,
    gc_enabled: bool = True,
) -> Tuple[List[Any], Dict[Any, Any]]:
    """Get pointers to args and kwargs.

    Args:
        args: List of arguments.
        kwargs: Dict of Keyword arguments.
        client: Client node.

    Returns:
        Tuple of args and kwargs with pointer to values.
    """
    # When we try to send params to a remote function they need to be pointers so
    # that they can be serialized and fetched from the remote store on arrival
    # this ensures that any args which are passed in from the user side are first
    # converted to pointers and sent then the pointer values are used for the
    # method invocation
    obj_lst = []
    pointer_args = []
    pointer_kwargs = {}
    for arg in args:
        # check if its already a pointer
        if not isinstance(arg, Pointer):
            arg_ptr, obj = arg.send(client, pointable=not gc_enabled, immediate=False)
            obj_lst.append(obj)
            pointer_args.append(arg_ptr)
        else:
            pointer_args.append(arg)
            arg.gc_enabled = gc_enabled

    for k, arg in kwargs.items():
        # check if its already a pointer
        if not isinstance(arg, Pointer):
            arg_ptr, obj = arg.send(client, pointable=not gc_enabled, immediate=False)
            obj_lst.append(obj)
            pointer_kwargs[k] = arg_ptr
        else:
            pointer_kwargs[k] = arg

    if obj_lst:
        msg = ActionSequence(obj_lst=obj_lst, address=client.address)

        # send message to client
        client.send_immediate_msg_without_reply(msg=msg)

    return pointer_args, pointer_kwargs
