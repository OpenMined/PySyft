import torch
from syft.exceptions import RemoteTensorFoundError
from .tensors import PointerTensor


def build_hook_args_function(args):
    # Inspect the call to find tensor arguments and return a rule whose
    # structure is the same as the args object, with 1 where there was
    # (torch or syft) tensors and 0 when not (ex: number, str, ...)
    rule = build_rule(args)
    # Build a function with this rule to efficiently replace syft tensors
    # (but not pointer) with their child in the args objects
    args_hook_function = build_args_hook(args, rule)
    return args_hook_function


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
    one = lambda _args: 1
    # dict to specify the action depending of the type found
    type_rule = {
        list: lambda _args: [build_rule(a) for a in _args],
        tuple: lambda _args: tuple([build_rule(a) for a in _args]),
        PointerTensor: one,
        torch.Tensor: one,
    }
    type_args = type(args)
    if type_args in type_rule:
        return type_rule[type_args](args)
    else:
        return 0


def build_args_hook(args, rules, return_tuple=False):
    """
    Build a function given some rules to efficiently replace in the args object
    syft tensors (but not pointer) with their child, and do nothing for other
    type of object including torch tensors, str, numbers, bool, etc
    Pointers trigger an error which is catched to get the location for forwarding
    the call
    :param args:
    :param rules:
    :param return_tuple: force to return a tuple even with a single element
    :return:
    """
    # Dict to return the proper lambda function for the right torch or syft tensor type
    forward_func = {
        PointerTensor: lambda p: (_ for _ in ()).throw(RemoteTensorFoundError(p)),
        torch.Tensor: lambda i: i.child if hasattr(i, "child") else i,
        "my_syft_tensor_type": lambda i: i.child,
    }
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
