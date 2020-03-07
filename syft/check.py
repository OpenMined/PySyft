from dpcontracts import require, ensure


def type_hints(decorated):
    # """If __debug__ is set to true, enforce python argument type hints
    #
    #   As a convention across PySyft,
    #
    #   Arg:
    #       decorated (func): the function on which we want to check argument types
    #
    #   Returns:
    #       decorator (func): the decorated function which checks argument types
    #   """

    def check_argument_types(args):

        for arg_name, arg_type in decorated.__annotations__.items():
            try:
                arg_value = getattr(args, arg_name)
            except AttributeError as e:
                raise AttributeError(
                    f"'{arg_name}' was passed into a function as an arg instead of a kwarg. Please pass in arguments as kwargs."
                )

            if not isinstance(arg_value, arg_type):
                raise AttributeError(
                    f"Argument '{arg_name}' should be an {arg_type} but received {type(arg_value)} instead"
                )

        return True

    @require("Wrong argument type", check_argument_types)
    def decorator(*args, **kwargs):
        return decorated(*args, **kwargs)

    return decorator


def max_one_arg(decorated):
    @require(
        "Please use args for at most 1 variable, kwargs for all other variables. Note that 'self' is counted as 1 variable.",
        lambda args: len(args.args) <= 1,
    )
    def decorator(*args, **kwargs):
        return decorated(*args, **kwargs)

    return decorator


def kwargs_only_function(decorated):
    @require(
        "Please only use kwargs when calling this function",
        lambda args: len(args.args) == 0,
    )
    def decorator(*args, **kwargs):
        return decorated(*args, **kwargs)

    return decorator


def kwargs_only_method(decorated):
    @require(
        "Please only use kwargs when calling this method",
        lambda args: len(args.args) <= 1,
    )
    def decorator(*args, **kwargs):
        return decorated(*args, **kwargs)

    return decorator
