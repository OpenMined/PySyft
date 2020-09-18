from typing import Union

import syft as sy
from syft.workers.abstract import AbstractWorker
from syft.generic.abstract.syft_serializable import SyftSerializable
from syft_proto.execution.v1.type_wrapper_pb2 import NestedTypeWrapper as NestedTypeWrapperPB
from syft_proto.execution.v1.type_wrapper_pb2 import InputTypeDescriptor as InputTypeDescriptorPB


class NestedTypeWrapper(SyftSerializable):
    """
    Class for input type serialization and type checking for nested structures.
    """

    def __init__(self, nested_type=None):
        self.nested_input_types = NestedTypeWrapper.enumerate_nested_types(nested_type)

    @staticmethod
    def get_object_identifiers(obj: any) -> (str, str):
        """
        Looks for identifiers for different objects, currently, only plans are supported
        with `name`, other identifiers can be added as well, eg.: `id`.

        Params:
            ojb: the object that you are typechecking

        Returns:
            (str, str): a tuple containing the type name and and unique str to identify that object.
        """
        type_name = type(obj).__name__

        if hasattr(obj, "name"):
            object_name = obj.name
        else:
            object_name = repr(obj)

        return (type_name, object_name)

    @staticmethod
    def enumerate_nested_types(input_arg: any) -> Union[list, tuple, dict, type]:
        """
        Method to enumerate the input of a function/Plan, including nested types.

        Note: supported nested structures: list, tuple, dict with string keys.

        Params:
            input_arg: *args of a function or Plan.

        Returns:
            Union[list, tuple, dict, type]: Nested structured with types instead of objects.
        """
        if input_arg is None:
            return None

        if isinstance(input_arg, (list, tuple)):
            result = [NestedTypeWrapper.enumerate_nested_types(elem) for elem in input_arg]
            return tuple(result) if isinstance(input_arg, tuple) else result

        if isinstance(input_arg, dict):
            serialized_dict = {
                k: NestedTypeWrapper.enumerate_nested_types(v) for k, v in input_arg.items()
            }
            return serialized_dict

        return type(input_arg)

    @staticmethod
    def raise_typecheck_err(typechecked_object: any, build: str, call: str, path: str) -> None:
        """
        Function to raise a type error if two types differ.

        Params:
            obj_type: the type of the object returned by calling .__name__ on it.
            obj_name: the name/id of the object.
            build: the build/reference argument type.
            call: the called argument type.
            path: the nested path to reach that obj.

        Returns:
             None
        """
        type_name, obj_name = NestedTypeWrapper.get_object_identifiers(typechecked_object)
        raise TypeError(
            f"{type_name} {obj_name} {path} has type {build}, while being built with type {call}.",
        )

    @staticmethod
    def raise_missmatch_err(typechecked_object: any, build: int, call: int, path: str) -> None:
        """
        Function to raise an error if two nested structures differ in length.

        Params:
            obj_type: the type of the object returned by calling .__name__ on it.
            obj_name: the name/id of the object.
            build: the build/reference argument length.
            call: the called argument length.
            path: the nested path to reach that obj.

        Returns:
             None
        """
        type_name, obj_name = NestedTypeWrapper.get_object_identifiers(typechecked_object)
        raise TypeError(
            f"{type_name} {obj_name} {path} has length {call}, "
            f"while being build with length {build}."
        )

    @staticmethod
    def raise_wrong_number_arguments_err(typechecked_object: any, build: int, call: int) -> None:
        """
        Function to raise an error if the build/reference function has a different number
        of arguments.

        Params:
            obj_type: the type of the object returned by calling .__name__ on it.
            obj_name: the name/id of the object.
            build: the build/reference input length.
            call: the called input length.

        Returns:
             None
        """
        type_name, obj_name = NestedTypeWrapper.get_object_identifiers(typechecked_object)
        raise TypeError(f"{type_name} {obj_name} requires {build} arguments, received {call}.")

    @staticmethod
    def raise_key_missing_err(typechecked_object: any, key: any, path: str) -> None:
        """
        Function to raise an error if the build/reference function has a different number
        of arguments.

        Params:
            obj_type: the type of the object returned by calling .__name__ on type(obj).
            obj_name: the name/id of the object.
            key: the key that is missing from the called dict.
            path: the nested path to reach that obj.

        Returns:
             None
        """
        type_name, obj_name = NestedTypeWrapper.get_object_identifiers(typechecked_object)
        raise KeyError(
            f"{type_name} {obj_name} {path} does not provide the key {key}, "
            "while being build with that key."
        )

    def input_check(self, typechecked_object: any, args: list) -> None:
        """
        Method for input validation by comparing the serialized build input with the
        current call input, following the following steps:
            1. Input length validation - checking that build and call inputs match on length.
            2. Verify the following nested structures: list, tuple, dict recursively. Lengths
            must match when comparing two nested lists, tuples or dicts. If they differ, an
            error will be raised.
            3. If we hit an object for which we don't support nesting, we compare types between
            call input and build input. If they differ, a warning will be raised.
            4. Dicts on the same nesting level on build and call input must have the same keys.
            If they differ, an error will be raised.

        Params:
            obj_type: the type of the object returned by calling .__name__ on type(obj).
            obj_name: the name/id of the object
            args: the arguments to be compared with the reference/build one.

        Returns:
            None
        """

        def check_type_nested_structure(
            typechecked_object,
            build_arg_nested_type: Union[list, tuple, dict, type],
            call_arg_nested_obj: any,
            path: str,
        ) -> None:
            """
            Recursive method to compare the nested input argument and the nested build argument.

            Params:
                build_arg_nested_type: Can be either a nested element (list, tuple, dict)
                                    or a type.
                call_arg_nested_obj: Can be either a nested element (list, tuple, dict)
                                    or an object.

            Returns:
                None
            """

            iterable_supported_list = (list, tuple, dict)

            if type(call_arg_nested_obj) not in iterable_supported_list:
                if not isinstance(call_arg_nested_obj, build_arg_nested_type):
                    NestedTypeWrapper.raise_typecheck_err(
                        typechecked_object,
                        build_arg_nested_type.__name__,
                        type(call_arg_nested_obj).__name__,
                        path,
                    )
                return

            if type(build_arg_nested_type) != type(call_arg_nested_obj):
                NestedTypeWrapper.raise_typecheck_err(
                    typechecked_object,
                    type(build_arg_nested_type).__name__,
                    type(call_arg_nested_obj).__name__,
                    path,
                )
                return

            if isinstance(build_arg_nested_type, (list, tuple)):
                if len(build_arg_nested_type) != len(call_arg_nested_obj):
                    NestedTypeWrapper.raise_missmatch_err(
                        typechecked_object,
                        len(build_arg_nested_type),
                        len(call_arg_nested_obj),
                        path,
                    )

                for idx in range(len(build_arg_nested_type)):
                    check_type_nested_structure(
                        typechecked_object,
                        build_arg_nested_type[idx],
                        call_arg_nested_obj[idx],
                        f"element {idx} of " + path,
                    )

            if isinstance(build_arg_nested_type, dict):
                if len(build_arg_nested_type) != len(call_arg_nested_obj):
                    NestedTypeWrapper.raise_missmatch_err(
                        typechecked_object,
                        len(build_arg_nested_type),
                        len(call_arg_nested_obj),
                        path,
                    )

                for key in build_arg_nested_type.keys():
                    if key in call_arg_nested_obj:
                        check_type_nested_structure(
                            typechecked_object,
                            build_arg_nested_type[key],
                            call_arg_nested_obj[key],
                            f"key {key} of " + path,
                        )
                    else:
                        NestedTypeWrapper.raise_key_missing_err(typechecked_object, key, path)

        if len(args) != len(self.nested_input_types):
            NestedTypeWrapper.raise_wrong_number_arguments_err(
                typechecked_object, len(self.nested_input_types), len(args)
            )

        for idx in range(len(args)):
            check_type_nested_structure(
                typechecked_object,
                self.nested_input_types[idx],
                args[idx],
                f"element {idx} of input",
            )

    @staticmethod
    def simplify(worker: AbstractWorker, nested_type_wrapper: "NestedTypeWrapper") -> list:
        return sy.serde.msgpack.serde._simplify(worker, nested_type_wrapper.nested_input_types)

    @staticmethod
    def detail(worker: AbstractWorker, simplified_nested_type: list) -> "NestedTypeWrapper":
        nested_type_wrapper = sy.serde.msgpack.serde._detail(worker, simplified_nested_type)
        result = NestedTypeWrapper()
        result.nested_input_types = nested_type_wrapper
        return result

    @staticmethod
    def bufferize(
        worker: AbstractWorker, nested_type_wrapper: "NestedTypeWrapper"
    ) -> NestedTypeWrapperPB:
        def bufferize_nested_structure(worker: AbstractWorker, obj: any) -> NestedTypeWrapperPB:
            nested_type_pb = NestedTypeWrapperPB()

            if isinstance(obj, list):
                container = NestedTypeWrapperPB.TypeContainer()
                proto_list = NestedTypeWrapperPB.TypeList()

                for elem in obj:
                    proto_list.nested_types.append(bufferize_nested_structure(worker, elem))

                container.nested_type_list.CopyFrom(proto_list)
                nested_type_pb.nested_types.CopyFrom(container)

            if isinstance(obj, tuple):
                container = NestedTypeWrapperPB.TypeContainer()
                proto_tuple = NestedTypeWrapperPB.TypeTuple()

                for elem in obj:
                    proto_tuple.nested_types.append(bufferize_nested_structure(worker, elem))

                container.nested_type_tuple.CopyFrom(proto_tuple)
                nested_type_pb.nested_types.CopyFrom(container)

            if isinstance(obj, dict):
                container = NestedTypeWrapperPB.TypeContainer()
                proto_map = NestedTypeWrapperPB.TypeMap()

                for k, v in obj.items():
                    proto_map.nested_types[k].CopyFrom(bufferize_nested_structure(worker, v))

                container.nested_type_dict.CopyFrom(proto_map)
                nested_type_pb.nested_types.CopyFrom(container)

            if isinstance(obj, type):
                container = NestedTypeWrapperPB.TypeContainer()
                typePB = InputTypeDescriptorPB()
                module_path = obj.__module__
                full_path_type = module_path + "." + obj.__name__
                typePB.type_name = full_path_type
                container.nested_type.CopyFrom(typePB)
                nested_type_pb.nested_types.CopyFrom(container)
            return nested_type_pb

        result = bufferize_nested_structure(worker, nested_type_wrapper.nested_input_types)
        return result

    @staticmethod
    def unbufferize(worker: AbstractWorker, message):
        def unbufferize_nested_structure(worker, message):
            container = None
            if message.nested_types.HasField("nested_type"):
                return sy.serde.protobuf.serde._unbufferize(
                    worker, message.nested_types.nested_type
                )

            if message.nested_types.HasField("nested_type_list"):
                container = []
                for obj in message.nested_types.nested_type_list.nested_types:
                    container.append(unbufferize_nested_structure(worker, obj))

            if message.nested_types.HasField("nested_type_tuple"):
                container = []
                for obj in message.nested_types.nested_type_tuple.nested_types:
                    container.append(unbufferize_nested_structure(worker, obj))
                container = tuple(container)

            if message.nested_types.HasField("nested_type_dict"):
                container = {}
                for k, v in message.nested_types.nested_type_dict.nested_types.items():
                    container[k] = unbufferize_nested_structure(worker, v)

            return container

        result = unbufferize_nested_structure(worker, message)
        wrapper = NestedTypeWrapper()
        wrapper.nested_input_types = result
        return wrapper

    @staticmethod
    def get_protobuf_schema():
        return NestedTypeWrapperPB
