import syft as sy
from syft.workers.abstract import AbstractWorker
from warnings import warn
from typing import Union
from syft_proto.execution.v1.type_wrapper_pb2 import NestedTypeWrapper as NestedTypeWrapperPB


class NestedTypeWrapper:
    """
        Class for input type serialization and type checking for nested structures.
    """
    def __init__(self, nested_type=None):
        if nested_type:
            self.serialized_nested_type = NestedTypeWrapper.serialize_nested_type(nested_type)
        else:
            self.serialized_nested_type = None

    @staticmethod
    def serialize_nested_type(input_arg: any) -> Union[list, tuple, dict, type]:
        """
            Function to serialize input of a function/Plan, including nested types.

            Note: supported nested structures: list, tuple, dict with string keys.

            Params:
                input_arg: *args of a function or Plan.

            Returns:
                Union[list, tuple, dict, type]: Nested structured with types instead of objects.
        """
        if isinstance(input_arg, (list, tuple)):
            result = [NestedTypeWrapper.serialize_nested_type(elem) for elem in input_arg]
            return tuple(result) if isinstance(input_arg, tuple) else result

        if isinstance(input_arg, dict):
            serialized_dict = {k: NestedTypeWrapper.serialize_nested_type(v) for k, v in input_arg.items()}
            return serialized_dict

        return type(input_arg)

    @staticmethod
    def raise_typecheck_warn(obj_type: str, obj_name: str, build: str, call: str, path: str) -> None:
        """
            Function to raise a typecheck warning if two types differ.

            Params:
                obj_type: the type of the object returned by calling .__name__ on it.
                obj_name: the name/id of the object.
                build: the build/reference argument type.
                call: the called argument type.
                path: the nested path to reach that obj.

            Returns:
                 None
        """
        warn(
            f"{obj_type} {obj_name} {path} has type {build}, while being built with type {call}.",
            RuntimeWarning,
        )

    @staticmethod
    def raise_missmatch_err(obj_type: str, obj_name: str, build: int, call: int, path: str) -> None:
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
        raise TypeError(f"{obj_type} {obj_name} {path} has length {call}, while being build with length {build}.")

    @staticmethod
    def raise_wrong_no_arguments_err(obj_type: str, obj_name: str, build: int, call: int) -> None:
        """
            Function to raise an error if the build/reference function has a different number of arguments.

            Params:
                obj_type: the type of the object returned by calling .__name__ on it.
                obj_name: the name/id of the object.
                build: the build/reference input length.
                call: the called input length.

            Returns:
                 None
        """
        raise TypeError(f"{obj_type} {obj_name} requires {build} arguments, received {call}.")

    @staticmethod
    def raise_key_missing_err(obj_type: str, obj_name: str, key: any, path: str) -> None:
        """
            Function to raise an error if the build/reference function has a different number of arguments.

            Params:
                obj_type: the type of the object returned by calling .__name__ on type(obj).
                obj_name: the name/id of the object.
                key: the key that is missing from the called dict.
                path: the nested path to reach that obj.

            Returns:
                 None
        """
        raise KeyError(f"{obj_type} {obj_name} {path} does not provide the key {key}, while being build with that key.")

    def input_check(self, obj_type: str, obj_name: str, args: list) -> None:
        """
            Method for input validation by comparing the serialized build input with the
            current call input, following the following steps:
                1. Input length validation - checking that build and call inputs match on length.
                2. Verify the following nested structures: list, tuple, dict recursively. Lengths must match when
                comparing two nested lists, tuples or dicts. If they differ, an error will be raised.
                3. If we hit an object for which we don't support nesting, we compare types between call input and
                build input. If they differ, a warning will be raised.
                4. Dicts on the same nesting level on build and call input must have the same keys. If they differ, an
                error will be raised.

            Params:
                obj_type: the type of the object returned by calling .__name__ on type(obj).
                obj_name: the name/id of the object
                args: the arguments to be compared with the reference/build one.

            Returns:
                None
        """
        def check_type_nested_structure(obj_type: str, obj_name: str, build: any, call: any, path: str) -> None:
            iterable_supported_list = (list, tuple, dict)

            if type(call) not in iterable_supported_list:
                if not isinstance(call, build):
                    NestedTypeWrapper.raise_typecheck_warn(obj_type, obj_name, build.__name__, type(call).__name__, path)
                return

            if type(build) != type(call):
                NestedTypeWrapper.raise_typecheck_warn(obj_type, obj_name, type(build).__name__, type(call).__name__, path)
                return

            if isinstance(build, (list, tuple)):
                if len(build) != len(call):
                    NestedTypeWrapper.raise_missmatch_err(obj_type, obj_name, len(build), len(call), path)

                for idx in range(len(build)):
                    check_type_nested_structure(obj_type, obj_name, build[idx], call[idx], f"element {idx} of " + path)

            if isinstance(build, dict):
                if len(build) != len(call):
                    NestedTypeWrapper.raise_missmatch_err(obj_type, obj_name, len(build), len(call), path)

                for key in build.keys():
                    if key in call:
                        check_type_nested_structure(
                            obj_type, obj_name, build[key], call[key], f"key {key} of " + path
                        )
                    else:
                        NestedTypeWrapper.raise_key_missing_err(obj_type, obj_name, key, path)

        if len(args) != len(self.serialized_nested_type):
            NestedTypeWrapper.raise_wrong_no_arguments_err(obj_type, obj_name, len(self.serialized_nested_type), len(args))

        for idx in range(len(args)):
            check_type_nested_structure(
                obj_type, obj_name, self.serialized_nested_type[idx], args[idx], f"element {idx} of input"
            )

    @staticmethod
    def simplify(worker: AbstractWorker, nested_type_wrapper: "NestedTypeWrapper") -> list:
        return sy.serde.msgpack.serde._simplify(worker, nested_type_wrapper.serialized_nested_type)

    @staticmethod
    def detail(worker: AbstractWorker, simplified_nested_type: list) -> "NestedTypeWrapper":
        nested_type_wrapper = sy.serde.msgpack.serde._detail(worker, simplified_nested_type)
        result = NestedTypeWrapper()
        result.serialized_nested_type = nested_type_wrapper
        return result

    @staticmethod
    def bufferize(worker: AbstractWorker, nested_type_wrapper: "NestedTypeWrapper") -> NestedTypeWrapperPB:
        def bufferize_nested_structure(worker: AbstractWorker, obj: any) -> NestedTypeWrapperPB:
            nested_type_pb = NestedTypeWrapperPB()

            if isinstance(obj, list):
                _ = [nested_type_pb.nested_type_list.append(bufferize_nested_structure(worker, elem)) for elem in obj]

            if isinstance(obj, tuple):
                _ = [nested_type_pb.nested_type_tuple.append(bufferize_nested_structure(worker, elem)) for elem in obj]

            if isinstance(obj, dict):
                for k, v in obj.items():
                    if not isinstance(k, str):
                        raise NotImplementedError("Plans support at input only dicts with string keys only at this moment.")
                    key_value_message = NestedTypeWrapperPB.key_value()
                    key_value_message.key = k.encode("utf-8")
                    key_value_message.value.CopyFrom(bufferize_nested_structure(worker, v))

                    nested_type_pb.nested_type_dict.append(key_value_message)

            if isinstance(obj, type):
                msg = sy.serde.protobuf.serde._bufferize(worker, obj)
                nested_type_pb.nested_type.id = msg.id
                nested_type_pb.nested_type.type = msg.type
            return nested_type_pb

        result = bufferize_nested_structure(worker, nested_type_wrapper.serialized_nested_type)
        return result

    @staticmethod
    def unbufferize(worker: AbstractWorker, obj):
        def unbufferize_nested_structure(worker, obj):
            if obj.nested_type_list:
                nested_type_list = []
                for elem in obj.nested_type_list:
                    nested_type_list.append(unbufferize_nested_structure(worker, elem))
                return nested_type_list

            if obj.nested_type_tuple:
                nested_type_tuple = []
                for elem in obj.nested_type_tuple:
                    nested_type_tuple.append(unbufferize_nested_structure(worker, elem))
                return tuple(nested_type_tuple)

            if obj.nested_type_dict:
                nested_type_dict = {}
                for elem in obj.nested_type_dict:
                    key = elem.key.decode("utf-8")
                    value = unbufferize_nested_structure(worker, elem.value)
                    nested_type_dict[key] = value
                return nested_type_dict

            if obj.nested_type:
                return sy.serde.protobuf.serde._unbufferize(worker, obj.nested_type)

        result = unbufferize_nested_structure(worker, obj)
        wrapper = NestedTypeWrapper()
        wrapper.serialized_nested_type = result
        return wrapper

    def __iter__(self):
        return iter(self.serialized_nested_type)

    def __eq__(self, other):
        return self.serialized_nested_type == other
