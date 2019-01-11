import torch
from syft.exceptions import RemoteTensorFoundError
from .tensors import PointerTensor
from .tensors import LogTensor
from .tensors import TorchTensor

hook_method_args_functions = {}
hook_method_response_functions = {}

one = lambda _args: 1

# dict to specify the action depending of the type found
type_rule = {
    list: lambda _args: [build_rule(a) for a in _args],
    tuple: lambda _args: tuple([build_rule(a) for a in _args]),
    LogTensor: one,
    PointerTensor: one,
    torch.Tensor: one,
}

# Dict to return the proper lambda function for the right torch or syft tensor type
forward_func = {
    PointerTensor: lambda p: (_ for _ in ()).throw(RemoteTensorFoundError(p)),
    torch.Tensor: lambda i: i.child if hasattr(i, "child") else i,
    LogTensor: lambda i: i.child,
    "my_syft_tensor_type": lambda i: i.child,
}

# Dict to return the proper lambda function for the right torch or syft tensor type
backward_func = {
    TorchTensor: lambda i: i.wrap(),
    torch.Tensor: lambda i: i.wrap(),
    PointerTensor: lambda i: i,
    LogTensor: lambda i: LogTensor().on(i, wrap=False),
    "my_syft_tensor_type": lambda i: "my_syft_tensor_type().on(i)",
}


def hook_method_args(attr, method_self, args):
    """Method arguments are sometimes simple types (such as strings or ints) but
    sometimes they are custom Syft tensors such as wrappers (torch.Tensor) or LogTensor
    or some other tensor type. Complex types (which have a .child attribute) need to
    have arguments converted from the arg to arg.child so that the types match as the
    method is being called down the chain. To make this efficient, we cache which args
    need to be replaced with their children in a dictionary called
    hook_method_args_functions. However, sometimes a method (an attr) has multiple
    different argument signatures, such that sometimes arguments have .child objects
    and other times they don't (such as x.div(), which can accept either a tensor or a
    float as an argument). This invalidates the cache, so we need to have a try/except
    which refreshes the cache if the signature triggers an error.

    Args:
        attr (str): the name of the method being called
        method_self: the tensor on which the method is being called
        args (list): the arguments being passed to the tensor
    """
    # Specify an id to distinguish methods from different classes
    # TODO: analyse exactly the role of adding the type of self in the id
    # TODO: and the need to recalculate also the rule (should be the same)
    attr_id = type(method_self).__name__ + "." + attr

    try:
        # Load the utility function to transform the args
        hook_args = hook_method_args_functions[attr_id]
        # Try running it
        new_self, new_args = hook_args((method_self, args))

    except (IndexError, KeyError):  # Update the function in case of an error
        args_hook_function = build_hook_args_function((method_self, args))
        # Store this utility function in the registry
        hook_method_args_functions[attr_id] = args_hook_function
        # Run it
        new_self, new_args = args_hook_function((method_self, args))

    return (new_self, new_args)


def build_hook_args_function(args):
    # Inspect the call to find tensor arguments and return a rule whose
    # structure is the same as the args object, with 1 where there was
    # (torch or syft) tensors and 0 when not (ex: number, str, ...)
    rule = build_rule(args)
    # Build a function with this rule to efficiently replace syft tensors
    # (but not pointer) with their child in the args objects
    args_hook_function = build_args_hook(args, rule)
    return args_hook_function


def hook_method_response(attr, response, wrap_type):
    """
    When executing a command, arguments are inspected and all tensors are replaced
    with their child attribute until a pointer or a torch tensor is found (for
    example an argument could be a torch wrapper with a child being a LogTensor, with
    a child being a torch tensor). When the result of the command is calculated,
    we need to rebuild this chain in the reverse order (in our example put back
    a LogTensor on top of the result and then a torch wrapper).
    To make this efficient, we cache which elements of the response (which can be more
    complicated with nested tuples for example) need to be wrapped in a dictionary called
    hook_method_response_functions. However, sometimes a method (an attr) has multiple
    different response signatures. This invalidates the cache, so we need to have a
    try/except which refreshes the cache if the signature triggers an error.

    Args:
        attr (str): the name of the method being called
        response (list): the arguments being passed to the tensor
        wrap_type (type): the type of wrapper we'd like to have
    """
    # TODO: Why do we need to cast it in a tuple? this is a (small) time waste
    response_is_tuple = isinstance(response, tuple)

    # Add an artificial tuple
    if not response_is_tuple:
        response = (response, 1)

    try:
        # Load the utility function to transform the args
        response_hook_function = hook_method_response_functions[attr]
        # Try running it
        new_response = response_hook_function(response)

    except (IndexError, KeyError):  # Update the function in cas of an error
        response_hook_function = build_hook_response_function(response, wrap_type)
        # Store this utility function in the registry
        hook_method_args_functions[attr] = response_hook_function
        # Run it
        new_response = response_hook_function(response)

    # Remove the artificial tuple
    if not response_is_tuple:
        new_response, _ = new_response

    return new_response


def build_hook_response_function(response, wrap_type):
    # Inspect the call to find tensor arguments and return a rule whose
    # structure is the same as the response object, with 1 where there was
    # (torch or syft) tensors and 0 when not (ex: number, str, ...)
    rule = build_rule(response)
    # Build a function with this rule to efficiently replace syft tensors
    # (but not pointer) with their child in the args objects
    response_hook_function = build_response_hook(response, rule, wrap_type)
    return response_hook_function


def build_rule(args):
    """
    Inspect the args object to find torch or syft tensor arguments and
    return a rule whose structure is the same as the args object,
    with 1 where there was (torch or syft) tensors and 0 when
    not (ex: number, str, ...)

    Example:
        in: ([tensor(1, 2), Pointer@bob], 42)
        out: ([1, 1], 0)
    """

    type_args = type(args)
    if type_args in type_rule:
        return type_rule[type_args](args)
    else:
        return 0


def build_args_hook(args, rules, return_tuple=False):
    """
    Build a function given some rules to efficiently replace in the args object
    syft tensors with their child (but not pointer as they don't have .child),
    and do nothing for other type of object including torch tensors, str,
    numbers, bool, etc.
    Pointers trigger an error which can be caught to get the location for
    forwarding the call.
    :param args:
    :param rules:
    :param return_tuple: force to return a tuple even with a single element
    :return: a function that replace syft arg in args with arg.child
    """

    # get the transformation lambda for each args
    lambdas = [
        (lambda i: i)  # return the same object
        if not r  # if the rule is a number == 0.
        else build_args_hook(a, r, True)  # If not, call recursively build_args_hook
        if isinstance(r, (list, tuple))  # if the rule is a list or tuple.
        # Last if not, rule is probably == 1 so use type to return the right transformation.
        else lambda i: forward_func[type(i)](i)
        for a, r in zip(args, rules)  # And do this for all the args / rules provided
    ]

    # Instead of iterating which is slow, we use trick to efficiently
    # apply each lambda to each arg
    folds = {0: zero_fold, 1: one_fold(return_tuple), 2: two_fold, 3: three_fold, 4: four_fold}
    f = folds[len(lambdas)]
    return lambda x: f(lambdas, x)


def build_response_hook(response, rules, wrap_type, return_tuple=False):
    """
    Build a function given some rules to efficiently replace in the response object
    syft or torch tensors with a wrapper, and do nothing for other types of object
    including , str, numbers, bool, etc.
    :param response:
    :param rules:
    :param return_tuple: force to return a tuple even with a single element
    :return:
    """

    # get the transformation lambda for each args
    lambdas = [
        (lambda i: i)  # return the same object
        if not r  # if the rule is a number == 0.
        else build_response_hook(
            a, r, wrap_type, True
        )  # If not, call recursively build_response_hook
        if isinstance(r, (list, tuple))  # if the rule is a list or tuple.
        # Last if not, rule is probably == 1 so use type to return the right transformation.
        else lambda i: backward_func[wrap_type](i)
        for a, r in zip(response, rules)  # And do this for all the responses / rules provided
    ]

    # Instead of iterating which is slow, we use trick to efficiently
    # apply each lambda to each arg
    folds = {0: zero_fold, 1: one_fold(return_tuple), 2: two_fold, 3: three_fold, 4: four_fold}
    f = folds[len(lambdas)]
    return lambda x: f(lambdas, x)


def zero_fold(*a):
    return tuple()


def one_fold(return_tuple):
    def _one_fold(lambdas, args):
        return lambdas[0](args[0])

    def tuple_one_fold(lambdas, args):
        return (lambdas[0](args[0]),)

    return {False: _one_fold, True: tuple_one_fold}[return_tuple]


def two_fold(lambdas, args):
    return lambdas[0](args[0]), lambdas[1](args[1])


def three_fold(lambdas, args):
    return lambdas[0](args[0]), lambdas[1](args[1]), lambdas[2](args[2])


def four_fold(lambdas, args):
    return (lambdas[0](args[0]), lambdas[1](args[1]), lambdas[2](args[2]), lambdas[3](args[3]))
