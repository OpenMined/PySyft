import syft as sy
from syft.workers.abstract import AbstractWorker
from warnings import warn

from syft_proto.execution.v1.type_wrapper_pb2 import NestedTypeWrapper as NestedTypeWrapperPB


class NestedTypeWrapper:
    def __init__(self, nested_type = None):
        if nested_type:
            self.serialized_nested_type = NestedTypeWrapper.serialize_nested_type(nested_type)
        else:
            self.serialized_nested_type = None

    @staticmethod
    def serialize_nested_type(input_arg):
        if isinstance(input_arg, (list, tuple)):
            result = []
            for arg in input_arg:
                result.append(NestedTypeWrapper.serialize_nested_type(arg))

            if isinstance(input_arg, tuple):
                return tuple(result)
            else:
                return result

        if isinstance(input_arg, dict):
            serialized_dict = {}
            for k, v in input_arg.items():
                serialized_dict[k] = NestedTypeWrapper.serialize_nested_type(v)
            return serialized_dict

        return type(input_arg)

    @staticmethod
    def raise_typecheck_warn(
            obj_type: str, obj_name: str, build_arg_type: str, call_arg_type: str, nested_structure_path: str
    ) -> None:
        warn(
            f"{obj_type} {obj_name} {nested_structure_path} has type {build_arg_type}, while being built with type {call_arg_type}.",
            RuntimeWarning,
        )

    @staticmethod
    def raise_missmatch_err(
            obj_type: str, obj_name: str, build_arg_length: int, call_arg_length: int, nested_structure_path: str
    ) -> None:
        raise TypeError(
            f"{obj_type} {obj_name} {nested_structure_path} has length {call_arg_length}, while being build with length {build_arg_length}.",
        )

    @staticmethod
    def raise_wrong_no_arguments_err(obj_type: str, obj_name: str, build_length: int, call_length: int) -> None:
        raise TypeError(
            f"{obj_type} {obj_name} requires {build_length} arguments, received {call_length}."
        )

    @staticmethod
    def raise_key_missing_err(obj_type: str, obj_name: str, key: any, nested_structure_path: str) -> None:
        raise KeyError(
            f"{obj_type} {obj_name} {nested_structure_path} does not provide the key {key}, while being build with that key."
        )

    def input_check(self, obj_type: str, obj_name: str, args: list) -> None:
        """
            Method for input validation by comparing the serialized build input (self.serialized_input) with the
            current call input, following the following steps:
                1. Input length validation - checking that build and call inputs match on length.
                2. Verify the following nested structures: list, tuple, dict recursively. Lengths must match when
                comparing two nested lists, tuples or dicts. If they differ, an error will be raised.
                3. If we hit an object for which we don't support nesting, we compare types between call input and
                build input. If they differ, a warning will be raised.
                4. Dicts on the same nesting level on build and call input must have the same keys. If they differ, an
                error will be raised.
        """
        def check_type_nested_structure(
            obj_type: str, obj_name, build_arg: any, call_arg: any, suffix: str
        ) -> None:
            iterable_supported_list = (list, tuple, dict)

            if type(call_arg) not in iterable_supported_list:
                if not isinstance(call_arg, build_arg):
                    NestedTypeWrapper.raise_typecheck_warn(obj_type, obj_name, build_arg.__name__, type(call_arg).__name__, suffix)
                return

            if type(build_arg) != type(call_arg):
                NestedTypeWrapper.raise_typecheck_warn(obj_type, obj_name, type(build_arg).__name__, type(call_arg).__name__, suffix)
                return

            if isinstance(build_arg, (list, tuple)):
                if len(build_arg) != len(call_arg):
                    NestedTypeWrapper.raise_missmatch_err(obj_type, obj_name, len(build_arg), len(call_arg), suffix)

                for idx in range(len(build_arg)):
                    check_type_nested_structure(
                        obj_type, obj_name, build_arg[idx], call_arg[idx], f"element {idx} of " + suffix
                    )

            if isinstance(build_arg, dict):
                if len(build_arg) != len(call_arg):
                    NestedTypeWrapper.raise_missmatch_err(obj_type, obj_name, len(build_arg), len(call_arg), suffix)

                for key in build_arg.keys():
                    if key in call_arg:
                        check_type_nested_structure(
                            obj_type, obj_name, build_arg[key], call_arg[key], f"key {key} of " + suffix
                        )
                    else:
                        NestedTypeWrapper.raise_key_missing_err(obj_type, obj_name, key, suffix)

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

    def __iter__(self):
        return iter(self.serialized_nested_type)

    def __eq__(self, other):
        return self.serialized_nested_type == other

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
