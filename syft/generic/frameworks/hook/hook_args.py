from typing import Callable
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np

from syft.generic.frameworks.types import FrameworkTensorType
from syft.workers.abstract import AbstractWorker

from syft import exceptions


hook_method_args_functions = {}
hook_method_response_functions = {}
get_tensor_type_functions = {}

base_types = {int, float, str, bool, bytes, bytearray, complex}

one = lambda _args: 1
get_child = lambda i: i.child

### Hook Args Registries ###
# If you have a type that will be fed into hooked functions, you must add it to
# these registeries using the public functions in the Registration Logic
# section below.
# WARNING: Do not attempt to manipulate them by hand. These registries should
#    not be used outside of this module. Use the helper functions instead.

# dict to specify the action depending of the type found
type_rule = {
    list: lambda _args: [build_rule(a) for a in _args],
    tuple: lambda _args: tuple(build_rule(a) for a in _args),
    dict: one,  # FIXME This is for additiveShareTensor.child, it can be confusing and AST.child
    np.ndarray: one,
    # should perhaps be of type ShareDict extending dict or something like this
}

# Dict to return the proper lambda function for the right framework or syft tensor type
forward_func = {"my_syft_tensor_type": get_child}

# Dict to return the proper lambda function for the right framework or syft tensor type
backward_func = {
    "my_syft_tensor_type": lambda i, **kwargs: "my_syft_tensor_type(**kwargs).on(i, wrap=False)"
}

# Methods or functions whose signature changes a lot and that we don't want to "cache", because
# they have an arbitrary number of tensors in args which can trigger unexpected behaviour
ambiguous_methods = set()
ambiguous_functions = {"run"}


### Registration logic ###
def register_type_rule(new_type_rules: Dict):
    global type_rule
    type_rule = {**type_rule, **new_type_rules}


def register_forward_func(new_forward_rules: Dict):
    global forward_func
    forward_func = {**forward_func, **new_forward_rules}


def register_backward_func(new_backward_rules: Dict):
    global backward_func
    backward_func = {**backward_func, **new_backward_rules}


def register_ambiguous_method(*method):
    global ambiguous_methods
    ambiguous_methods.update(set(method))


def register_ambiguous_function(*function):
    global ambiguous_functions
    ambiguous_functions.update(set(function))


def default_backward_func(tensorcls):
    return lambda i, **kwargs: tensorcls(**kwargs).on(i, wrap=False)


def default_register_tensor(*tensorcls):
    register_type_rule({t: one for t in tensorcls})
    register_forward_func({t: get_child for t in tensorcls})
    register_backward_func({t: default_backward_func(t) for t in tensorcls})


### Main hook args implementation ###


def unwrap_args_from_method(attr, method_self, args_, kwargs_):
    """Method arguments are sometimes simple types (such as strings or ints) but sometimes
    they are custom Syft tensors such as wrappers (i.e. FrameworkTensor), LoggingTensor
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
        args_ (list): the arguments being passed to the method
        kwargs_ (dict): the keyword arguments being passed to the function
            (these are not hooked ie replace with their .child attr)
    """
    # Specify an id to distinguish methods from different classes
    # As they won't be used with the same arg types
    attr_id = type(method_self).__name__ + "." + attr
    try:
        assert attr not in ambiguous_methods

        # Load the utility function to transform the args
        hook_args = hook_method_args_functions[attr_id]
        # Try running it
        new_self, new_args = hook_args((method_self, args_))

    except (IndexError, KeyError, AssertionError):  # Update the function in case of an error
        args_hook_function, _ = build_unwrap_args_from_function((method_self, args_))
        # Store this utility function in the registry
        hook_method_args_functions[attr_id] = args_hook_function
        # Run it
        new_self, new_args = args_hook_function((method_self, args_))

    return new_self, new_args, kwargs_


def unwrap_args_from_function(attr, args_, kwargs_, return_args_type=False):
    """See unwrap_args_from_method for details

    Args:
        attr (str): the name of the function being called
        args_ (list): the arguments being passed to the function
        kwargs_ (dict): the keyword arguments being passed to the function
            (these are not hooked ie replace with their .child attr)
        return_args_type (bool): return the type of the tensors in the
        original arguments

    Returns:
        - the arguments where all tensors are replaced with their child
        - the type of this new child
        (- the type of the tensors in the arguments)
    """
    try:
        assert attr not in ambiguous_functions
        # Load the utility function to transform the args
        # TODO rename registry or use another one than for methods
        hook_args = hook_method_args_functions[attr]
        get_tensor_type_function = get_tensor_type_functions[attr]

        # Try running it
        new_args = hook_args(args_)

    except (IndexError, KeyError, AssertionError):  # Update the function in case of an error
        args_hook_function, get_tensor_type_function = build_unwrap_args_from_function(
            args_, return_tuple=True
        )
        # Store the utility functions in registries
        hook_method_args_functions[attr] = args_hook_function
        get_tensor_type_functions[attr] = get_tensor_type_function
        # Run it
        new_args = args_hook_function(args_)

    new_type = get_tensor_type_function(new_args)
    if return_args_type:
        args_type = get_tensor_type_function(args_)
        return new_args, kwargs_, new_type, args_type
    else:
        return new_args, kwargs_, new_type


def build_unwrap_args_from_function(args_, return_tuple=False):
    """
    Build the function f that hook the arguments:
    f(args_) = new_args
    """
    # Inspect the call to find tensor arguments and return a rule whose
    # structure is the same as the args_ object, with 1 where there was
    # (framework or syft) tensors and 0 when not (ex: number, str, ...)
    rule = build_rule(args_)
    # Build a function with this rule to efficiently replace syft tensors
    # (but not pointer) with their child in the args_ objects
    args_hook_function = build_unwrap_args_with_rules(args_, rule, return_tuple)
    # Build a function with this rule to efficiently the child type of the
    # tensor found in the args_
    get_tensor_type_function = build_get_tensor_type(rule)
    return args_hook_function, get_tensor_type_function


def hook_response(attr, response, wrap_type, wrap_args={}, new_self=None):
    """
    When executing a command, arguments are inspected and all tensors are replaced
    with their child attribute until a pointer or a framework tensor is found (for
    example an argument could be a framework wrapper with a child being a LoggingTensor, with
    a child being a framework tensor). When the result of the command is calculated,
    we need to rebuild this chain in the reverse order (in our example put back
    a LoggingTensor on top of the result and then a framework wrapper).
    To make this efficient, we cache which elements of the response (which can be more
    complicated with nested tuples for example) need to be wrapped in a dictionary called
    hook_method_response_functions. However, sometimes a method (an attr) has multiple
    different response signatures. This invalidates the cache, so we need to have a
    try/except which refreshes the cache if the signature triggers an error.

    Args:
        attr (str): the name of the method being called
        response (list or dict): the arguments being passed to the tensor
        wrap_type (type): the type of wrapper we'd like to have
        wrap_args (dict): options to give to the wrapper (for example the
        precision for the precision tensor)
        new_self: used for the can just below of inplace ops
    """

    # inline methods should just return new_self
    if "__i" == attr[0:3] and attr != "__iter__":
        return new_self

    # TODO: Why do we need to cast it in a tuple? this is a (small) time waste
    response_is_tuple = isinstance(response, tuple)

    # Add an artificial tuple
    if not response_is_tuple:
        response = (response, 1)

    hash_wrap_args = hash(frozenset(wrap_args.items()))
    attr_id = f"{attr}@{wrap_type.__name__}.{response_is_tuple}.{hash_wrap_args}"

    try:
        assert attr not in ambiguous_functions

        # Load the utility function to transform the args
        response_hook_function = hook_method_response_functions[attr_id]
        # Try running it
        new_response = response_hook_function(response)

    except (IndexError, KeyError, AssertionError):  # Update the function in case of an error
        response_hook_function = build_wrap_response_from_function(response, wrap_type, wrap_args)
        # Store this utility function in the registry
        hook_method_response_functions[attr_id] = response_hook_function
        # Run it
        new_response = response_hook_function(response)

    # Remove the artificial tuple
    if not response_is_tuple:
        new_response, _ = new_response

    return new_response


def build_wrap_response_from_function(response, wrap_type, wrap_args):
    """
    Build the function that hook the response.

    Example:
        p is of type Pointer
        f is the hook_response_function
        then f(p) = (Wrapper)>Pointer
    """
    # Inspect the call to find tensor arguments and return a rule whose
    # structure is the same as the response object, with 1 where there was
    # (framework or syft) tensors and 0 when not (ex: number, str, ...)
    rule = build_rule(response)
    # Build a function with this rule to efficiently replace syft tensors
    # (but not pointer) with their child in the args objects
    response_hook_function = build_wrap_response_with_rules(response, rule, wrap_type, wrap_args)
    return response_hook_function


def build_rule(args_):
    """
    Inspect the args object to find framework or syft tensor arguments and
    return a rule whose structure is the same as the args object,
    with 1 where there was (framework or syft) tensors and 0 when
    not (ex: number, str, ...)

    Example:
        in: ([tensor(1, 2), Pointer@bob], 42)
        out: ([1, 1], 0)
    """

    type_args = type(args_)
    # for list, tuple but also tensors and syft tensors
    if type_args in type_rule:
        return type_rule[type_args](args_)
    # for int, float, str, etc
    elif type_args in base_types:
        return 0
    else:
        # New kind of return with pytorch 1.1
        if "torch.return_types" in str(type_args):
            return type_rule[tuple](args_)
        # Still remain ellipsis, slices, etc.
        return 0


def build_unwrap_args_with_rules(args_, rules, return_tuple=False, return_list=False):
    """
    Build a function given some rules to efficiently replace in the args object
    syft tensors with their child (but not pointer as they don't have .child),
    and do nothing for other type of object including framework tensors, str,
    numbers, bool, etc.
    Pointers trigger an error which can be caught to get the location for
    forwarding the call.

    Args:
        args_ (tuple): the arguments given to the function / method
        rules (tuple): the same structure but with boolean, true when there is
            a tensor
        return_tuple (bool): force to return a tuple even with a single element
        return_list (bool): force to return a list instead of a tuple

    Return:
        a function that replace syft arg in args_ with arg.child
    """

    # get the transformation lambda for each args
    lambdas = [
        typed_identity(a)  # return the same obj with an identity fct with a type check if needed
        if not r  # if the rule is a number == 0.
        else build_unwrap_args_with_rules(a, r, True, True)
        if isinstance(r, list)
        else build_unwrap_args_with_rules(
            a, r, True
        )  # If not, call recursively build_unwrap_args_with_rules
        if isinstance(r, tuple)
        # Last if not, rule is probably == 1 so use type to return the right transformation.
        else lambda i: forward_func[type(i)](i)
        for a, r in zip(args_, rules)  # And do this for all the args / rules provided
    ]

    # Instead of iterating which is slow, we use trick to efficiently
    # apply each lambda to each arg
    folds = {
        0: zero_fold,
        1: one_fold(return_tuple),
        2: two_fold,
        3: three_fold,
        4: four_fold,
        5: five_fold,
        6: six_fold,
        7: seven_fold,
        8: eight_fold,
    }
    try:
        f = folds[len(lambdas)]
    except KeyError:
        f = many_fold

    if return_list:
        return lambda x: list(f(lambdas, x))

    return lambda x: f(lambdas, x)


def build_get_tensor_type(rules, layer=None):
    """
    Build a function which uses some rules to find efficiently the first tensor in
    the args objects and return the type of its child.

    Args:
        rules (tuple): a skeleton object with the same structure as args but each tensor
            is replaced with a 1 and other types (int, str) with a 0
        layer (list or None): keep track of the path of inspection: each element in the list
            stand for one layer of deepness into the object, and its value for the index
            in the current layer. See example for details

    Returns:
        a function returning a type

    Example:
        *Understanding the layer parameter*
        obj = (a, [b, (c, d)], e)
        the layer position is for:
        a: [0]
        b: [1, 0]
        c: [1, 1, 0]
        d: [1, 1, 1]
        e: [2]

        *Global behaviour example*
        rules = (0, [1, (0, 0), 0)
        - First recursion level
          0 found -> do nothing
          list found -> recursive call with layer = [1]
        - Second recursion level
          1 found -> update layer to [1, 0]
                     build the function x: type(x[1][0])
                     break
        - Back to first recursion level
          save the function returned in the lambdas list
          0 found -> do nothing
          exit loop
          return the first (and here unique) function


    """
    # We keep note of the first layer or recursion level to return at the end
    # only one function and instantiate the layer list the first time
    first_layer = layer is None

    if first_layer:
        layer = []

    # Iteration through the rules object
    lambdas = []
    for i, r in enumerate(rules):
        if r == 1:  # if a tensor is found
            layer.append(i)
            lambdas.append(
                # the layer object is given to build a getter to reach the
                # tensor position and then the type() is called on the obj found
                lambda a: type(get_element_at[len(layer)](*layer)(a))
            )
            # we only need one to get the type of all tensors as they should be the same
            break
        if isinstance(r, (list, tuple)):  # we iterate recursively if necessary
            layer.append(i)
            lambdas += build_get_tensor_type(r, layer)

    if first_layer:
        try:
            return lambdas[0]
        except IndexError:
            # Some functions don't have tensors in their signature so rules is only made of 0s,
            # Hence lambdas is empty. Raising PureFrameworkTensorFoundError triggers an execution of
            # the un-hooked (so native) function which is perfect in that case.
            raise exceptions.PureFrameworkTensorFoundError
    else:
        return lambdas


# Function helpers to convert [a, b, c, ...] -> obj[a][b][c][...]
def one_layer(idx1):
    return lambda l: l[idx1]


def two_layers(idx1, idx2):
    return lambda l: one_layer(idx2)(l[idx1])


def three_layers(idx1, *ids):
    return lambda l: two_layers(*ids)(l[idx1])


def four_layers(idx1, *ids):
    return lambda l: three_layers(*ids)(l[idx1])


get_element_at = {1: one_layer, 2: two_layers, 3: three_layers, 4: four_layers}


def build_wrap_response_with_rules(
    response, rules, wrap_type, wrap_args, return_tuple=False, return_list=False
):
    """
    Build a function given some rules to efficiently replace in the response object
    syft or framework tensors with a wrapper, and do nothing for other types of object
    including , str, numbers, bool, etc.

    Args:
        response: a response used to build the hook function
        rules: the same structure objects but with boolean, at true when is replaces
            a tensor
        return_tuple: force to return a tuple even with a single element
        return_list: force to return a list instead of a tuple

    Response:
        a function to "wrap" the response
    """

    # get the transformation lambda for each args
    lambdas = [
        (lambda i: i)  # return the same object
        if not r  # if the rule is a number == 0.
        else build_wrap_response_with_rules(a, r, wrap_type, wrap_args, True, True)
        if isinstance(r, list)
        else build_wrap_response_with_rules(
            a, r, wrap_type, wrap_args, True
        )  # If not, call recursively build_wrap_response_with_rules
        if isinstance(r, tuple)
        # Last if not, rule is probably == 1 so use type to return the right transformation.
        else lambda i: backward_func[wrap_type](i, **wrap_args)
        for a, r in zip(response, rules)  # And do this for all the responses / rules provided
    ]

    # Instead of iterating which is slow, we use trick to efficiently
    # apply each lambda to each arg
    folds = {
        0: zero_fold,
        1: one_fold(return_tuple),
        2: two_fold,
        3: three_fold,
        4: four_fold,
        5: five_fold,
        6: six_fold,
        7: seven_fold,
        8: eight_fold,
    }
    try:
        f = folds[len(lambdas)]
    except KeyError:
        f = many_fold

    if return_list:
        return lambda x: list(f(lambdas, x))

    return lambda x: f(lambdas, x)


def zero_fold(*a, **k):
    return ()


def one_fold(return_tuple, **kwargs):
    def _one_fold(lambdas, args_, **kwargs):
        return lambdas[0](args_[0], **kwargs)

    def tuple_one_fold(lambdas, args_):
        return (lambdas[0](args_[0], **kwargs),)

    return {False: _one_fold, True: tuple_one_fold}[return_tuple]


def two_fold(lambdas, args_, **kwargs):
    return lambdas[0](args_[0], **kwargs), lambdas[1](args_[1], **kwargs)


def three_fold(lambdas, args_, **kwargs):
    return (
        lambdas[0](args_[0], **kwargs),
        lambdas[1](args_[1], **kwargs),
        lambdas[2](args_[2], **kwargs),
    )


def four_fold(lambdas, args_, **kwargs):
    return (
        lambdas[0](args_[0], **kwargs),
        lambdas[1](args_[1], **kwargs),
        lambdas[2](args_[2], **kwargs),
        lambdas[3](args_[3], **kwargs),
    )


def five_fold(lambdas, args_, **kwargs):
    return (
        lambdas[0](args_[0], **kwargs),
        lambdas[1](args_[1], **kwargs),
        lambdas[2](args_[2], **kwargs),
        lambdas[3](args_[3], **kwargs),
        lambdas[4](args_[4], **kwargs),
    )


def six_fold(lambdas, args_, **kwargs):
    return (
        lambdas[0](args_[0], **kwargs),
        lambdas[1](args_[1], **kwargs),
        lambdas[2](args_[2], **kwargs),
        lambdas[3](args_[3], **kwargs),
        lambdas[4](args_[4], **kwargs),
        lambdas[5](args_[5], **kwargs),
    )


def seven_fold(lambdas, args_, **kwargs):
    return (
        lambdas[0](args_[0], **kwargs),
        lambdas[1](args_[1], **kwargs),
        lambdas[2](args_[2], **kwargs),
        lambdas[3](args_[3], **kwargs),
        lambdas[4](args_[4], **kwargs),
        lambdas[5](args_[5], **kwargs),
        lambdas[6](args_[6], **kwargs),
    )


def eight_fold(lambdas, args_, **kwargs):
    return (
        lambdas[0](args_[0], **kwargs),
        lambdas[1](args_[1], **kwargs),
        lambdas[2](args_[2], **kwargs),
        lambdas[3](args_[3], **kwargs),
        lambdas[4](args_[4], **kwargs),
        lambdas[5](args_[5], **kwargs),
        lambdas[6](args_[6], **kwargs),
        lambdas[7](args_[7], **kwargs),
    )


def many_fold(lambdas, args_, **kwargs):
    return tuple(lambdas[i](args_[i], **kwargs) for i in range(len(lambdas)))


# Add the possibility to make a type check in the identity function applied
# On some arg which could be None are of another type.
# Could add more checks but not sure it is needed so far.


def typed_identity(a):
    """
    We need to add typed identity for arguments which can be either number
    or tensors. If the argument changes from an int to a tensor, the
    assertion error triggered by typed_identity will be caught and a
    new signature will be computed for the command.
    """
    if a is None:

        def none_identity(i):
            assert i is None
            return i

        return none_identity

    elif type(a) in (int, float, bool):

        def number_identity(i):
            assert isinstance(i, type(a))
            return i

        return number_identity

    else:
        return lambda i: i


# -- Fast way to register responses and transform tensors in pointers

register_response_functions = {}


def register_response(
    attr: str, response: object, response_ids: object, owner: AbstractWorker
) -> object:
    """
    When a remote worker execute a command sent by someone else, the response is
    inspected: all tensors are stored by this worker and a Pointer tensor is
    made for each of them.

    To make this efficient, we cache which elements of the response (which can be more
    complicated with nested tuples for example) in the dict register_response_functions

    However, sometimes a function  (an attr) has multiple different response signatures.
    This invalidates the cache, so we need to have a try/except which refreshes the
    cache if the signature triggers an error.

    Args:
        attr (str): the name of the function being called
        response (object): the response of this function
        owner (BaseWorker): the worker which registers the tensors
    """

    # TODO: Why do we need to cast it in a tuple? this is a (small) time waste
    response_is_tuple = isinstance(response, tuple)

    # Add an artificial tuple
    if not response_is_tuple:
        response = (response, 1)

    attr_id = f"{attr}"

    try:
        assert attr not in ambiguous_functions
        assert attr not in ambiguous_methods

        # Load the utility function to register the response and transform tensors with pointers
        register_response_function = register_response_functions[attr_id]
        # Try running it
        new_response = register_response_function(response, response_ids=response_ids, owner=owner)

    except (IndexError, KeyError, AssertionError):  # Update the function in cas of an error
        register_response_function = build_register_response_function(response)
        # Store this utility function in the registry
        register_response_functions[attr_id] = register_response_function
        # Run it
        new_response = register_response_function(response, response_ids=response_ids, owner=owner)

    # Remove the artificial tuple
    if not response_is_tuple:
        new_response, _ = new_response

    return new_response


def build_register_response_function(response: object) -> Callable:
    """
    Build the function that registers the response and replaces tensors with pointers.

    Example:
        (1, tensor([1, 2]) is the response
        f is the register_response_function
        then f(p) = (1, (Wrapper)>Pointer)
    """
    # Inspect the call to find tensor arguments and return a rule whose
    # structure is the same as the response object, with 1 where there was
    # (framework or syft) tensors and 0 when not (ex: number, str, ...)
    rule = build_rule(response)
    # Build a function with this rule to efficiently replace syft tensors
    # (but not pointer) with their child in the args_ objects
    response_hook_function = build_register_response(response, rule)
    return response_hook_function


def register_tensor(tensor: FrameworkTensorType, owner: AbstractWorker, response_ids: List = []):
    """
    Registers a tensor.

    Args:
        tensor: A tensor.
        owner: The owner that makes the registration.
        response_ids: List of ids where the tensor should be stored
            and each id is pop out when needed.
    """
    # This method often leads to re-registration of tensors
    # hence creating two copies of the same info. The older tensor
    # is left hanging and is never deleted. De-Registering the original
    # tensor (if-exists) before registration addresses this problem.
    owner.de_register_obj(tensor)  # Doesn't raise Exceptions if absent on owner
    tensor.owner = owner
    try:
        tensor.id = response_ids.pop(0)
    except IndexError:
        raise exceptions.ResponseSignatureError

    owner.register_obj(tensor)

    return tensor


def build_register_response(response: object, rules: Tuple, return_tuple: bool = False) -> Callable:
    """
    Build a function given some rules to efficiently replace in the response object
    framework tensors with a pointer after they are registered, and do nothing for other
    types of object including , str, numbers, bool, etc.

    Args:
        response: the response
        rules: the rule specifying where the tensors are
        return_tuple: force to return a tuple even with a single element
    Returns:
        The function to apply on generic responses
    """

    # get the transformation lambda for each args_
    lambdas = [
        (lambda i, **kwargs: i)  # return the same object
        if not r  # or not hasattr(a, "owner")  # if the rule is a number == 0.
        else build_register_response(
            a, r, True
        )  # If not, call recursively build_wrap_response_with_rules
        if isinstance(r, (list, tuple))  # if the rule is a list or tuple.
        # Last if not, rule is probably == 1 so use type to return the right transformation.
        else lambda i, **kwargs: register_tensor(i, **kwargs)
        for a, r in zip(response, rules)  # And do this for all the responses / rules provided
    ]

    # Instead of iterating which is slow, we use trick to efficiently
    # apply each lambda to each arg
    folds = {
        0: zero_fold,
        1: one_fold(return_tuple),
        2: two_fold,
        3: three_fold,
        4: four_fold,
        5: five_fold,
        6: six_fold,
        7: seven_fold,
        8: eight_fold,
    }
    try:
        f = folds[len(lambdas)]
    except KeyError:
        f = many_fold

    return lambda x, **kwargs: f(lambdas, x, **kwargs)
