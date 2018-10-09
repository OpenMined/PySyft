"""Torch static utility functions."""
import json
import re
from enum import IntEnum
import types
import functools
import numpy as np
import logging
import torch
import copy
import syft
import syft as sy

from .. import encode
from ... import utils


def extract_type_and_obj(dct):
    """Utils function, which given a serialized tensors, Returns a pair tuple
    with the tensor type (in a string) and the associated data in a dict."""
    pat = re.compile("__(.+)__")
    for key, obj in dct.items():
        if pat.search(key) is not None:
            obj_type = pat.search(key).group(1)
            return obj_type, obj
        else:
            raise TypeError("Key", key, "is not recognized.")


def get_child_command(obj, child_types=[]):
    """Analyse a Python object (generally dict with a command and arguments,
    And for all tensors, variables, syft tensors, replace with their first
    child and retrieve its type.

    :param obj:
    :param child_types:
    :return:
    """
    # Torch tensor or variable, or sy._SyftTensor
    if not isinstance(obj, str) and (
        is_tensor(obj) or is_variable(obj) or is_syft_tensor(obj)
    ):
        obj_type = type(obj.child)
        # We identify Parameter type with Variable type since they are quite close
        # TODO: What are the risks due to this assimilation? (see usage @ torch/utils.py l.74)
        if obj_type is sy.Parameter:
            obj_type = sy.Variable
        return obj.child, [obj_type]
    # List or iterables which could contain tensors
    elif isinstance(obj, (list, tuple, set, bytearray, range)):
        children = []
        types = []
        for o in obj:
            c, t = get_child_command(o, child_types)
            children.append(c)
            types += t
        return type(obj)(children), types
    # Dict
    elif isinstance(obj, dict):
        children = {}
        types = []
        for k, o in obj.items():
            c, t = get_child_command(o, child_types)
            children[k] = c
            types += t
        return children, types
    else:
        return obj, []


def prepare_child_command(command, replace_tensorvar_with_child=False):
    """Returns a command where all tensors are replaced with their child, and
    returns also the type For now we expect all the children to share the same
    type."""
    next_command, next_child_types = get_child_command(command)

    # Check that the next child type of all tensorvar is the same
    # TODO: should allow to mix Variable and Parameter in next_child_types
    n_types = len(next_child_types)
    if n_types == 0:
        ref_child_type = sy._LocalTensor
    elif n_types == 1:
        ref_child_type = next_child_types[0]
    else:
        if all(child_type in torch.tensorvar_types_tuple for child_type in next_child_types):
            ref_child_type = next_child_types[0]
        else:
            ref_child_type = next_child_types[0]
            for next_child_type in next_child_types:
                if next_child_type != ref_child_type:

                    if "self" in next_command:
                        assert_tensors_on_same_machine(
                            list(next_command["args"]) + [next_command["self"]]
                        )
                    else:
                        assert_tensors_on_same_machine(next_command["args"])

                    raise NotImplementedError(
                        "All arguments should share the same child type.",
                        next_child_types,
                    )

    if replace_tensorvar_with_child:
        return next_command, ref_child_type
    else:
        return command, ref_child_type


def assert_tensors_on_same_machine(tensors):
    def fail_msg(a, b):
        raise Exception(
            "Tensors must be on the same machine - {0} != {1}. Move them to the same machine using"
            "my_tensor.send()",
            a,
            b,
        )

    reference_tensor = tensors[0]

    if isinstance(reference_tensor, sy._PointerTensor):
        for tensor in tensors[1:]:
            if isinstance(tensor, sy._PointerTensor):
                if tensor.location != reference_tensor.location:
                    fail_msg(tensor.location, reference_tensor.location)

            elif isinstance(tensor, sy._GeneralizedPointerTensor):
                pt_gpt_error(reference_tensor, tensor)
            elif isinstance(tensor.child, sy._PointerTensor):
                raise Exception(
                    "All tensors must have the same class hierarchy. Your tensor with id:"
                    + tensor.child.id
                    + " might line up except the PointerTensor object is wrapped"
                    + "with a tensor of type "
                    + str(type(tensor))
                )

            elif isinstance(tensor, sy._LocalTensor):
                pt_loc_error(reference_tensor, tensor)

    elif isinstance(reference_tensor, sy._GeneralizedPointerTensor):
        for tensor in tensors[1:]:
            if isinstance(tensor, sy._GeneralizedPointerTensor):
                if (
                    tensor.pointer_tensor_dict.keys()
                    != reference_tensor.pointer_tensor_dict.keys()
                ):
                    msg = "All pointers must point to the same group of workers!\n\n"
                    msg += str(tensor.pointer_tensor_dict.keys())
                    msg += "\n is not the same as\n\n"
                    msg += str(reference_tensor.pointer_tensor_dict.keys())
                    raise Exception(msg)

    elif isinstance(reference_tensor, sy._LocalTensor):

        for tensor in tensors[1:]:
            if isinstance(tensor, sy._PointerTensor):
                pt_loc_error(tensor, reference_tensor)

    return True


def pt_loc_error(ptr, local):
    raise Exception(
        "All tensors must be on the same machine. \n\nYou just tried to call an operation"
        + " between a tensor on your local machine and a tensor on remote worker '"
        + str(ptr.location.id)
        + "'. Call .send(\""
        + str(ptr.location.id)
        + '") on the local tensor (the one with ID:'
        + str(local.id)
        + ") or .get() on the"
        + " remote tensor (the one with ID:"
        + str(ptr.id)
        + ")"
    )


def pt_gpt_error(ptr, gptr):
    raise Exception(
        "Tensors must be on the same machine. You just tried to perform an operation"
        "between a PointerTensor (which points to one machine: "
        + str(ptr.location)
        + ") and a GeneralizedPointerTensor"
        "(which points to several machines"
        + str(gptr.pointer_tensor_dict.keys())
        + "). Try "
        "sending your PointerTensor to all machines"
        "in the GeneralizedPointerTensor or vise versa. "
    )


def enforce_owner(obj, owner):
    """Reassign every elements of the chain to a specified owner (in a Virtual
    worker context)"""

    if isinstance(obj, (list, tuple, set, bytearray)):
        for o in obj:
            enforce_owner(o, owner)
    elif isinstance(obj, dict):
        for k, o in obj.items():
            enforce_owner(o, owner)

    else:
        if (
            is_syft_tensor(obj)
            and not isinstance(obj, sy._LocalTensor)
            and hasattr(obj, "data")
        ):
            enforce_owner(obj.data, owner)
        if (
            is_syft_tensor(obj)
            and not isinstance(obj, sy._LocalTensor)
            and hasattr(obj, "grad")
        ):
            enforce_owner(obj.grad, owner)

        if is_tensor(obj):
            if owner != owner.hook.local_worker:
                owner.hook.local_worker.de_register(obj)
            # tensor has no attr owner, just a prop to obj.child
            enforce_owner(obj.child, owner)

        elif is_variable(obj):
            if owner != owner.hook.local_worker:
                owner.hook.local_worker.de_register(obj)
            # tensor has no attr owner, just a prop to obj.child
            enforce_owner(obj.child, owner)
            enforce_owner(obj.data, owner)
            if obj.grad is not None:
                enforce_owner(obj.grad, owner)

        elif is_syft_tensor(obj):
            if owner != owner.hook.local_worker:
                owner.hook.local_worker.de_register(obj)
            obj.owner = owner
            # Terminal condition to avoid recursions
            if not isinstance(obj, sy._LocalTensor):
                enforce_owner(obj.child, owner)

        elif isinstance(obj, np.ndarray):
            if owner != owner.hook.local_worker:
                owner.hook.local_worker.de_register(obj)
            try:
                obj.owner = owner
            except:
                """sometimes this failes."""

            # would normally call enforce_owner(obj.child, owner) here except since
            # Torch is circular this creates an infinite recursion. TODO: fix after Torch 1.0


def bind_var_like_objects(obj, child_obj, grad=False):
    obj.child = child_obj
    child_obj.parent = obj

    obj.data.child = child_obj.data
    child_obj.data.parent = obj.data

    if grad:
        obj.grad.child = child_obj.grad
        child_obj.grad.parent = obj.grad

        obj.grad.data.child = child_obj.grad.data
        child_obj.grad.data.parent = obj.grad.data


def wrap_command_with(obj, wrapper):
    """Wrap a syft object with a given wrapper."""
    wrapper.child = obj
    obj.parent = wrapper
    return wrapper


def wrap_command(obj):
    """To a Syft command, add a torch wrapper Returns the wrapper."""
    # for numeric values for instance, don't add a wrapper
    if isinstance(obj, (int, float, bool, str, np.ndarray, slice)) or obj is None:
        return obj
    # Torch tensor or variable
    elif is_tensor(obj) or is_variable(obj):
        return obj  # the tensor is already wrapped
        # raise TypeError('Expecting syft tensors but got ' + str(type(obj)))
    # sy._SyftTensor
    elif is_syft_tensor(obj):
        _tail = find_tail_of_chain(obj)
        if isinstance(_tail, sy._LocalTensor):
            wrapper = _tail.child
        else:
            wrapper = torch.guard[obj.torch_type]()

        wrap_command_with(obj, wrapper)
        if is_variable(wrapper):
            if hasattr(obj, "data"):
                wrapper.data = wrap_command(obj.data)
            if hasattr(obj, "grad"):
                wrapper_grad = wrap_command(obj.grad)
                wrapper.assign_grad_(wrapper_grad)

        return wrapper
    # List or iterables which could contain tensors
    elif isinstance(obj, (list, tuple, set, bytearray, range)):
        return type(obj)([wrap_command(o) for o in obj])
    # Dict
    elif isinstance(obj, dict):
        return {k: wrap_command(o) for k, o in obj.items()}
    else:
        # print('The following type wasnt wrapped:', str(type(obj)))
        return obj


def get_connected_variables(variable):
    """Return all variables involved in the backward process, using the
    auxiliary get_variables_in_backward_graph function."""
    variables, _ = get_variables_in_backward_graph(variable.grad_fn, [], set())
    return variables


def get_variables_in_backward_graph(var, nodes=[], seen=set()):
    if var not in seen:
        if torch.is_tensor(var):
            logging.warning("Shouldnt access tensors")
        elif hasattr(var, "variable"):
            u = var.variable
            nodes.append(u.id)  # id(var), id(u)
        else:
            pass
        seen.add(var)
        if hasattr(var, "next_functions"):
            for u in var.next_functions:
                if u[0] is not None:
                    nodes, seen = get_variables_in_backward_graph(u[0], nodes, seen)
        if hasattr(var, "saved_tensors"):
            for t in var.saved_tensors:
                nodes, seen = get_variables_in_backward_graph(t, nodes, seen)
    return nodes, seen


def compile_command(attr, args, kwargs, has_self=False, self=None):
    """Returns the JSON serializable encoded command, the location which should
    receive it and the owners of the pointers seen in the command Is used in
    the _PointerTensor handlecall to prepare the command before forwarding it
    to a remote worker."""
    command = {"command": attr, "has_self": has_self, "args": args, "kwargs": kwargs}
    if has_self:
        command["self"] = self

    command, pointers = encode.encode(command, retrieve_pointers=True)

    # Get information about the location and owner of the pointers
    locations = set()
    owners = set()
    for pointer in pointers:
        locations.add(pointer.location)
        owners.add(pointer.owner)

    locations = list(locations)
    owners = list(owners)

    if len(locations) > 1:
        raise NotImplementedError("All pointers should point to the same worker")
    if len(owners) > 1:
        raise NotImplementedError("All pointers should share the same owner.")

    return command, locations, owners


def split_to_pointer_commands(syft_command):
    """Split a syft command containing _GeneralizedPointerTensor with n
    pointers in n syft commands, each for one pointer/worker.

    :param syft_command
    :return: n syft_commands
    """
    # TODO: See Issue #1480
    base_command = {
        "has_self": syft_command["has_self"],
        "command": syft_command["command"],
        "kwargs": syft_command["kwargs"],
        "args": [],
    }
    worker_ids = []
    syft_commands = {}

    if syft_command["has_self"]:
        if isinstance(syft_command["self"], sy._GeneralizedPointerTensor):
            for worker_id, pointer in syft_command["self"].pointer_tensor_dict.items():
                # Init phase >
                syft_commands[worker_id] = copy.deepcopy(base_command)
                worker_ids.append(worker_id)
                # < end
                syft_commands[worker_id]["self"] = pointer
        else:
            base_command["self"] = syft_command["self"]

    for arg in syft_command["args"]:
        if isinstance(arg, sy._GeneralizedPointerTensor):
            if len(syft_commands) == 0:
                for worker_id, pointer in arg.pointer_tensor_dict.items():
                    # Init phase >
                    syft_commands[worker_id] = copy.deepcopy(base_command)
                    worker_ids.append(worker_id)
                    # < end
            for worker_id, pointer in arg.pointer_tensor_dict.items():
                syft_commands[worker_id]["args"].append(pointer)
        elif isinstance(arg, list) and isinstance(arg[0], sy._GeneralizedPointerTensor):
            # this logic is supposed to handle hierarchical lists of generalizedpointertensors
            # with somewhat tested support for torch.cat
            if len(syft_commands) == 0:
                for worker_id, pointer in arg[0].pointer_tensor_dict.items():
                    # Init phase >
                    syft_commands[worker_id] = copy.deepcopy(base_command)
                    worker_ids.append(worker_id)

            arg_lists = {}
            for worker_id, pointer in arg[0].pointer_tensor_dict.items():
                arg_lists[worker_id] = []

            for _arg in arg:
                for worker_id, pointer in _arg.pointer_tensor_dict.items():
                    arg_lists[worker_id].append(pointer)

            for worker_id, arg_list in arg_lists.items():
                syft_commands[worker_id]["args"].append(arg_list)

        elif isinstance(arg, (list, set, tuple)):
            if len(syft_commands) == 0:
                base_command["args"] = arg
            else:
                for worker_id in worker_ids:
                    syft_commands[worker_id]["args"].append(arg)
            # raise NotImplementedError('Cant deal with nested args on Generalizd Pointers')
        else:
            if len(syft_commands) == 0:
                base_command["args"].append(arg)
            else:
                for worker_id in worker_ids:
                    syft_commands[worker_id]["args"].append(arg)

    return syft_commands


def assert_has_only_torch_tensorvars(obj):
    """A check function that an object has only torch Tensors or Variable at
    his 'roots', ie head of chain Is useful for development."""
    if isinstance(obj, (int, float, str, slice, type(...))):
        return True
    elif is_tensor(obj):
        return True
    elif is_variable(obj):
        return True
    elif isinstance(obj, (list, tuple)):
        rep = [assert_has_only_torch_tensorvars(o) for o in obj]
        return all(rep)
    elif isinstance(obj, dict):
        rep = [assert_has_only_torch_tensorvars(o) for o in obj.values()]
        return all(rep)
    elif callable(obj):
        return True
    elif obj is None:
        return True
    else:
        assert False, ("Obj is not tensorvar", type(obj))


def assert_has_only_syft_tensors(obj):
    """A check function that an object has only syft Tensors at his 'roots', ie
    head of chain Is useful for development."""
    if isinstance(obj, (int, float, str, slice, type(...))):
        return True
    elif issubclass(obj.__class__, sy._SyftTensor):
        return True
    elif isinstance(obj, (list, tuple)):
        rep = [assert_has_only_syft_tensors(o) for o in obj]
        return all(rep)
    elif isinstance(obj, dict):
        rep = [assert_has_only_syft_tensors(o) for o in obj.values()]
        return all(rep)
    elif callable(obj):
        return True
    elif obj is None:
        return True
    else:
        assert False, ("Obj is not syft tensor", type(obj))


def chain_print(obj, display=True, verbose=False):
    """Print the chain of a tensor or variable.

    Useful for debugging
    If not verbose:
        1st line = main chain
        (for variables only)
        2nd line = data chain
        3rd line = grad chain if any
    If verbose
        1st line = main chain
        (for variables only)
            2nd line = .data attr of all nodes of the main chain
        3rd line = data chain
            4th line = .grad attr of all nodes of the main chain
        5th line = grad chain if any
    """
    has_grad = False
    if is_variable(obj):
        is_var = True
        data_display = chain_print(obj.data, display=False, verbose=verbose)
        if obj.grad is not None:
            has_grad = True
            grad_display = chain_print(obj.grad, display=False, verbose=verbose)
    else:
        is_var = False
    if not verbose:
        types = [obj.__class__.__name__]
    else:
        types = [
            obj.__class__.__name__ + "(" + str(obj.owner.id) + ":" + str(obj.id) + ")"
        ]
        if is_var:
            data_types = [
                obj.data.__class__.__name__
                + "("
                + str(obj.data.owner.id)
                + ":"
                + str(obj.data.id)
                + ")"
            ]
            grad_types = [
                obj.grad.__class__.__name__
                + ("(" + str(obj.grad.owner.id) + ":" + str(obj.grad.id) + ")")
                if obj.grad is not None
                else ""
            ]
    i = 0
    while hasattr(obj, "child"):
        if not verbose:
            types.append(obj.child.__class__.__name__)
        else:
            types.append(
                obj.child.__class__.__name__
                + "("
                + str(obj.child.owner.id)
                + ":"
                + str(obj.child.id)
                + ")"
            )
            if is_var:
                if hasattr(obj.child, "data"):
                    data_types.append(
                        obj.child.data.__class__.__name__
                        + "("
                        + str(obj.child.data.owner.id)
                        + ":"
                        + str(obj.child.data.id)
                        + ")"
                    )
                else:
                    data_types.append("<empty>")

                if hasattr(obj.child, "grad"):
                    grad_types.append(
                        obj.child.grad.__class__.__name__
                        + (
                            "("
                            + str(obj.child.grad.owner.id)
                            + ":"
                            + str(obj.child.grad.id)
                            + ")"
                        )
                        if obj.child.grad is not None
                        else ""
                    )
                else:
                    grad_types.append("<empty>")
        if isinstance(obj.child, (sy._LocalTensor, sy._PointerTensor)):
            break
        if isinstance(obj.child, (sy._GeneralizedPointerTensor,)):
            break
        obj = obj.child
        i += 1
        if i >= 12:
            types.append("(...)")
            break
    if display:
        print(" > ".join(types))
        if verbose and is_var:
            print("[d]| " + " | ".join(data_types))
        if is_var:
            print(" - " + data_display)
        if verbose and is_var:
            print("[g]| " + " | ".join(grad_types))
        if has_grad:
            print(" - - " + "\n   - ".join(grad_display.split("\n - ")))
    else:
        display = " > ".join(types)
        if is_var:
            display += "\n - " + data_display
        if has_grad:
            display += "\n - - " + "\n   - ".join(grad_display.split("\n - "))
        return display


def get_syft_chain(obj):
    """Return the chain of syft object types in a list."""
    next_node = obj.child
    syft_chain = []
    while next_node is not None and not (
        is_tensor(next_node) or is_variable(next_node)
    ):
        syft_chain.append(type(next_node))
        next_node = next_node.child

    return syft_chain


def link_var_chain_to_data_chain(var_node, data_node):
    """Add a data attribute to all tensors involved in a Variable chain,
    pointing to the var.data chain, so that calls to .data and .child can be
    permuted."""
    var_node.data = data_node
    next_node = var_node.child
    if next_node is not None and not (is_tensor(next_node) or is_variable(next_node)):
        link_var_chain_to_data_chain(var_node.child, data_node.child)


def link_var_chain_to_data_and_grad_chains(var_node, data_node, grad_node):
    """Similar to link_var_chain_to_data_chain, but also add a .grad attribute
    pointing to the var.grad chain."""
    var_node.data = data_node
    if not is_variable(grad_node) or len(grad_node.size()) > 0:
        var_node.grad = grad_node
    next_node = var_node.child
    if (
        next_node is not None
    ):  # and not (is_tensor(next_node) or is_variable(next_node)):
        if isinstance(next_node, sy._LocalTensor):
            next_node.data = data_node.child
            if not is_variable(grad_node.child) or len(grad_node.child.size()) > 0:
                next_node.grad = grad_node.child
        else:
            link_var_chain_to_data_and_grad_chains(
                var_node.child, data_node.child, grad_node.child
            )


def assert_is_chain_well_formed(
    obj, downward=True, start_id=None, start_type=None, end_chain=None
):
    """
    Performs an analysis that a chain is correctly built:
    A local chain should be something that terminates with a _LocalTensor,
    e.g. `FloatTensor -> _LogTensor -> _LocalTensor`. In this setting the
    child and parent are obvious on the middle elements, and on the edge
    there is a "loop", _LocalTensor.child = FloatTensor and FloatTensor.parent
    = _LocalTensor.
    A non-local chain in something that terminates with a _PointerTensor for
    instance, e.g. `FloatTensor -> _LogTensor -> _PointerTensor`. In this case
    the edges of the chains shouldn't be connected because it makes no sense,
    and the remote protocol send/get/etc. is the equivalent of the child
    attribute which is missing for the pointer.

    In practice, we also check for unexpected loops.
    """
    # Is only executed at the first iteration
    if start_id is None:
        start_id = obj.id
        start_type = type(obj)
        if is_variable(obj):
            # We don't care about the return, as the main object has to return true anyway
            # All we care is about Exception raising
            assert_is_chain_well_formed(obj.data)
    else:
        if start_id == obj.id and start_type == type(obj):
            raise StopIteration(
                "The chain looped downward=",
                downward,
                "on id",
                obj.child.id,
                "with obj",
                obj.child,
            )
    if end_chain is not None and (is_variable(obj) or is_tensor(obj)):
        if isinstance(end_chain, sy._PointerTensor):
            assert (
                obj.parent is None
            ), "Tensorvar linked to Pointer should not have a parent"
            assert end_chain.child is None, "Pointer shouldnt have a child"
            return True
        elif isinstance(end_chain, sy._LocalTensor):
            # we allow to have inner tensors in the chain, provided that its parent is a FixedPTensor
            if not isinstance(obj.parent, sy._FixedPrecisionTensor):
                assert obj.parent.id == end_chain.id, (
                    "TensorVar parent should be the tail LocalTensor"
                    + str(obj.parent.id)
                    + ","
                    + str(end_chain.id)
                )
                assert (
                    end_chain.child.id == obj.id
                ), "Tail LocalTensor child should be the Tensor Var"
                return True

        elif isinstance(end_chain, sy._SPDZTensor):
            return True
        else:
            raise TypeError("Unsupported end_chain type:", obj)

    elif isinstance(obj, sy._PointerTensor):
        downward = False
        end_chain = obj
        start_id = obj.id
        start_type = type(obj)
    elif isinstance(obj, sy._LocalTensor):
        downward = False
        end_chain = obj
        start_id = obj.id
        start_type = type(obj)
    elif isinstance(obj, sy._SPDZTensor):
        downward = False
        end_chain = obj
        start_id = obj.id
        start_type = type(obj)

    if downward:
        if obj.child is None:
            raise AttributeError(
                "Chain broken downward without a Pointer at the end, but", obj
            )
        else:
            return assert_is_chain_well_formed(
                obj.child, downward, start_id, start_type, end_chain
            )
    else:
        if obj.parent is None:
            raise AttributeError("Chain broken upward, at", obj)
        else:
            return assert_is_chain_well_formed(
                obj.parent, downward, start_id, start_type, end_chain
            )


def find_tail_of_chain(obj, start_id=None, start_type=None):
    """Returns the last element of a chain, and perform basic sanity checks on
    the chain like unexpected loops."""
    if start_id is None:
        start_id = obj.id
        start_type = type(obj)
    else:
        if start_id == obj.id and start_type == type(obj):
            raise StopIteration(
                "The chain looped downward on id", obj.child.id, "with obj", obj.child
            )

    if isinstance(
        obj, (sy._LocalTensor, sy._PointerTensor, sy._GeneralizedPointerTensor)
    ):
        return obj
    else:
        if obj.child is None:
            raise AttributeError("Chain is broken on", obj)
        else:
            obj.child.parent = obj
            return find_tail_of_chain(obj.child, start_id, start_type)


def find_torch_object_in_family_tree(obj):
    """Returns the torch_objects on a local_chain (without pointer)"""
    ch = obj.child
    while True:
        if is_tensor(ch) or is_variable(ch):
            return ch
        if hasattr(ch, "child"):
            ch = ch.child
        else:
            raise AttributeError("Chain of", obj.id, "has Pointer or is broken on", ch)


def fix_chain_ends(obj):
    """Performs BASIC fixes on a chain, typically useful when decoding a JSON-
    ified chain to fix the child and parents attributes NOTE that this doesn't
    guarantee that the chain will be well-formed, so calling after
    `assert_is_chain_well_formed` will be a good idea."""
    end_obj = find_tail_of_chain(obj)
    if isinstance(end_obj, sy._LocalTensor):
        end_obj.child = obj
        obj.parent = end_obj
    elif isinstance(end_obj, (sy._PointerTensor, sy._GeneralizedPointerTensor)):
        end_obj.child = None
        obj.parent = None
    elif isinstance(end_obj, sy._SPDZTensor):
        """"""
    else:
        raise TypeError("Unsupported end of chain:", end_obj)

    if is_variable(obj):
        fix_chain_ends(obj.data)
        if obj.grad is not None:
            fix_chain_ends(obj.grad)


def is_tensor_empty(obj):
    # TODO Will break with PyTorch >= 0.4
    return obj.dim() == 0


def define_enums():
    """Define int encoding of usual string for fast type comparison."""
    # A global binding binding encoded types and integers
    supported_types = (
        ["worker"]
        + ["tuple", "set", "bytearray", "range"]
        + ["slice"]
        + torch.tensor_type_names
        + torch.var_type_names
        + torch.syft_tensor_name
    )

    normal_types = IntEnum("DynamicEnum", supported_types)
    torch.type_codes = normal_types

    # Add a conversion dictionary to remove __**__
    encoded_types = {"__" + t + "__": t for t in supported_types}
    torch.encoded_types = encoded_types

    # Add an enum for syft tensor
    syft_tensor_codes = {
        normal_types[syft_tensor] for syft_tensor in torch.syft_tensor_name
    }
    torch.syft_tensor_codes = syft_tensor_codes

    # Add an enum for tensor
    tensor_codes = {normal_types[tensor] for tensor in torch.tensor_type_names}
    torch.tensor_codes = tensor_codes

    # Add an enum for variable
    var_codes = {normal_types[variable] for variable in torch.var_type_names}
    torch.var_codes = var_codes


def type_code(type_name):
    try:
        return torch.type_codes[torch.encoded_types[type_name]]
    except KeyError:
        return -1


def is_syft_tensor(obj):
    """Determines whether the arg is a subclass of a SyftTensor or is the name
    of a subclass of a SyftTensor."""
    return issubclass(type(obj), sy._SyftTensor)


def is_syft_tensor_name(obj):
    """Determines whether the arg is a subclass of a SyftTensor or is the name
    of a subclass of a SyftTensor."""
    try:
        type_code = torch.type_codes[obj]
        return type_code in torch.syft_tensor_codes
    except KeyError:
        return False


def is_tensor(obj):
    """Determines whether the arg is a subclass of a Torch Tensor."""
    return isinstance(obj, torch.tensor_types_tuple)


def is_tensor_name(name):
    """Determines whether the arg is the name of a subclass of a Torch
    Tensor."""
    try:
        type_code = torch.type_codes[name]
        return type_code in torch.tensor_codes
    except KeyError:
        return False


def is_variable(obj):
    """Determines whether the arg is a Variable or is the (part of the) name of
    a class Variable."""
    return isinstance(obj, torch.var_types_tuple)


def is_variable_name(obj):
    """Determines whether the arg is the (part of the) name of a class
    Variable."""
    try:
        type_code = torch.type_codes[obj]
        return type_code in torch.var_codes
    except KeyError:
        return False
