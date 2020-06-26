from dpcontracts import require, ensure
import inspect
import typing
from collections.abc import Iterable, Sequence, Mapping

RETURN_TYPE_CHECK_IGNORE = {"__init__"}

SUBSCRIPTED_DTYPES = {list, set, tuple, Iterable, Sequence}

SUBSCRIPTED_KV_DTYPES = {dict, Mapping}


def type_hints(decorated):
    signature = inspect.signature(decorated)
    stack_message_error = None

    def check_return_type(result):
        if decorated.__name__ in RETURN_TYPE_CHECK_IGNORE:
            return True

        if signature.return_annotation is signature.empty:
            raise AttributeError(
                f"Return type not annotated, please provide typing to the return type."
            )

        return_type = (
            type(None)
            if signature.return_annotation is None
            else signature.return_annotation
        )
        if not isinstance(result, return_type):
            raise AttributeError(
                f"Return type is {type(result)}, should be {return_type}."
            )

        return True

    def generic_isinstance(local_arg, local_target_type, path):
        nonlocal stack_message_error

        generic_type = typing.get_origin(local_target_type)
        subscripted_type = typing.get_args(local_target_type)

        if generic_type is typing.Union:
            for nongeneric_type in subscripted_type:
                if generic_isinstance(local_arg, nongeneric_type, path):
                    return True
            stack_message_error = (
                path + f": Argument should have any of the types {subscripted_type}."
            )
            return False

        if generic_type:
            if not isinstance(local_arg, generic_type):
                stack_message_error = (
                    path + f": Argument should have type {generic_type}."
                )
                return False
        else:
            if not isinstance(local_arg, local_target_type):
                stack_message_error = (
                    path + f": Argument should have type {local_target_type}."
                )
                return False

        if generic_type in SUBSCRIPTED_DTYPES and subscripted_type:
            for elem in local_arg:
                if not generic_isinstance(elem, subscripted_type, path):
                    stack_message_error = (
                        path + f": Iterable should have type {local_target_type}."
                    )
                    return False

        if generic_type in SUBSCRIPTED_KV_DTYPES and subscripted_type:
            key_type = subscripted_type[0]
            value_type = subscripted_type[1]
            for k, v in local_arg.items():
                if not generic_isinstance(k, key_type, path):
                    stack_message_error = (
                        path + f": Key element of mapping should have type {key_type}."
                    )
                    return False

                if not generic_isinstance(v, value_type, path):
                    stack_message_error = (
                        path
                        + f"{path} Value element of mapping should have type {value_type}."
                    )
                    return False
            return True
        return True

    def check_argument_types(args):
        for idx, (param_name, param) in enumerate(signature.parameters.items()):
            if idx == 0 and param_name == "self":
                continue

            if param.annotation is param.empty:
                raise AttributeError(
                    f"{param_name} was passed into a function without the typing defined."
                )
            if hasattr(args, param_name):
                arg_value = getattr(args, param_name)
            elif param.default is not param.empty:
                arg_value = param.default
            else:
                raise AttributeError(
                    f"'{param_name}' was passed into a function as an arg instead of a kwarg."
                    f"Please pass in arguments as kwargs."
                )

            if not generic_isinstance(
                arg_value, param.annotation, path=f"Error in argument {param_name}"
            ):
                raise AttributeError(stack_message_error)
        return True

    @require("Wrong arg type", check_argument_types)
    @ensure("Wrong type", lambda args, result: check_return_type(result))
    def decorator(*args, **kwargs):
        return decorated(*args, **kwargs)

    return decorator
