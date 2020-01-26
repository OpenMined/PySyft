#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import crypten.communicator as comm
import crypten.mpc  # noqa: F401
import crypten.nn  # noqa: F401
import torch

# other imports:
from . import debug
from .cryptensor import CrypTensor
from .mpc import ptype


def init(rank=0, world_size=1):
    comm._init(use_threads=False,
               rank=rank,
               world_size=world_size,
               init_ttp=crypten.mpc.ttp_required()
            )
    if comm.get().get_rank() < comm.get().get_world_size():
        _setup_przs()
        if crypten.mpc.ttp_required():
            crypten.mpc.provider.ttp_provider.TTPClient._init()


def init_thread(rank, world_size):
    comm._init(use_threads=True, rank=rank, world_size=world_size)
    _setup_przs()


def uninit():
    return comm.uninit()


def is_initialized():
    return comm.is_initialized()


# the different private type attributes of an mpc encrypted tensor
arithmetic = ptype.arithmetic
binary = ptype.binary


def print_communication_stats():
    comm.get().print_communication_stats()


def reset_communication_stats():
    comm.get().reset_communication_stats()


# Set backend
__SUPPORTED_BACKENDS = [crypten.mpc]
__default_backend = __SUPPORTED_BACKENDS[0]


def set_default_backend(new_default_backend):
    """Sets the default cryptensor backend (mpc, he)"""
    global __default_backend
    assert new_default_backend in __SUPPORTED_BACKENDS, (
        "Backend %s is not supported" % new_default_backend
    )
    __default_backend = new_default_backend


def get_default_backend():
    """Returns the default cryptensor backend (mpc, he)"""
    return __default_backend


def cryptensor(*args, backend=None, **kwargs):
    """
    Factory function to return encrypted tensor of given backend.
    """
    if backend is None:
        backend = get_default_backend()
    if backend == crypten.mpc:
        return backend.MPCTensor(*args, **kwargs)
    else:
        raise TypeError("Backend %s is not supported" % backend)


def is_encrypted_tensor(obj):
    """
    Returns True if obj is an encrypted tensor.
    """
    return isinstance(obj, CrypTensor)


def _setup_przs():
    """
        Generate shared random seeds to generate pseudo-random sharings of
        zero. The random seeds are shared such that each process shares
        one seed with the previous rank process and one with the next rank.
        This allows for the generation of `n` random values, each known to
        exactly two of the `n` parties.

        For arithmetic sharing, one of these parties will add the number
        while the other subtracts it, allowing for the generation of a
        pseudo-random sharing of zero. (This can be done for binary
        sharing using bitwise-xor rather than addition / subtraction)
    """
    # Initialize RNG Generators
    comm.get().g0 = torch.Generator()
    comm.get().g1 = torch.Generator()

    # Generate random seeds for Generators
    # NOTE: Chosen seed can be any number, but we choose as a random 64-bit
    # integer here so other parties cannot guess its value.

    # We sometimes get here from a forked process, which causes all parties
    # to have the same RNG state. Reset the seed to make sure RNG streams
    # are different in all the parties. We use numpy's random here since
    # setting its seed to None will produce different seeds even from
    # forked processes.
    import numpy

    numpy.random.seed(seed=None)
    next_seed = torch.tensor(numpy.random.randint(-2 ** 63, 2 ** 63 - 1, (1,)))
    prev_seed = torch.LongTensor([0])  # placeholder

    # Send random seed to next party, receive random seed from prev party
    world_size = comm.get().get_world_size()
    rank = comm.get().get_rank()
    if world_size >= 2:  # Otherwise sending seeds will segfault.
        next_rank = (rank + 1) % world_size
        prev_rank = (next_rank - 2) % world_size

        req0 = comm.get().isend(tensor=next_seed, dst=next_rank)
        req1 = comm.get().irecv(tensor=prev_seed, src=prev_rank)

        req0.wait()
        req1.wait()
    else:
        prev_seed = next_seed

    # Seed Generators
    comm.get().g0.manual_seed(next_seed.item())
    comm.get().g1.manual_seed(prev_seed.item())


def __validate_model(loaded_model, dummy_model):
    """Validates that two models have the same architecture"""
    loaded_modules = [loaded_model]
    dummy_modules = [dummy_model]

    valid = torch.tensor(1, dtype=torch.long)
    try:
        while len(loaded_modules) > 0:
            loaded_module = loaded_modules.pop(0)
            dummy_module = dummy_modules.pop(0)

            # Assert modules have the same number of parameters
            loaded_params = [param for param in loaded_module.parameters()]
            dummy_params = [param for param in dummy_module.parameters()]
            assert len(loaded_params) == len(dummy_params)

            for i, param in enumerate(loaded_params):
                assert param.size() == dummy_params[i].size()

            # Assert that modules have the same number of sub-modules
            loaded_module_modules = [mod for mod in loaded_module.modules()][1:]
            dummy_module_modules = [mod for mod in dummy_module.modules()][1:]

            loaded_modules.extend(loaded_module_modules)
            dummy_modules.extend(dummy_module_modules)
            assert len(loaded_modules) == len(dummy_modules)
    except AssertionError:
        valid = torch.tensor(0, dtype=torch.long)
    return valid


def load(f, encrypted=False, dummy_model=None, src=0, **kwargs):
    """
    Loads an object saved with `torch.save()` or `crypten.save()`.

    Args:
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        encrypted: Determines whether crypten should load an encrypted tensor
                      or a plaintext torch tensor.
        dummy_model: Takes a model architecture to fill with the loaded model
                    (on the `src` party only). Non-source parties will return the
                    `dummy_model` input (with data unchanged). Loading a model will
                    assert the correctness of the model architecture provided against
                    the model loaded. This argument is ignored if the file loaded is
                    a tensor.
        src: Determines the source of the tensor. If `src` is None, each
            party will attempt to read in the specified file. If `src` is
            specified, the source party will read the tensor from
    """
    if encrypted:
        raise NotImplementedError("Loading encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Load failed: src argument must be an integer"
        assert (
            src >= 0 and src < comm.get().get_world_size()
        ), "Load failed: src must be in [0, world_size)"

        # TODO: Use send_obj and recv_obj to send modules without requiring a
        # dummy_model

        # source party
        if comm.get().get_rank() == src:
            result = torch.load(f, **kwargs)

            # file contains torch.tensor
            if torch.is_tensor(result):
                # Broadcast load type
                load_type = torch.tensor(0, dtype=torch.long)
                comm.get().broadcast(load_type, src=src)

                # Broadcast size to other parties.
                dim = torch.tensor(result.dim(), dtype=torch.long)
                size = torch.tensor(result.size(), dtype=torch.long)

                comm.get().broadcast(dim, src=src)
                comm.get().broadcast(size, src=src)
                result = cryptensor(result, src=src)

            # file contains torch module
            elif isinstance(result, torch.nn.Module):
                # Broadcast load type
                load_type = torch.tensor(1, dtype=torch.long)
                comm.get().broadcast(load_type, src=src)

                # Assert that dummy_model is provided
                assert dummy_model is not None and isinstance(
                    dummy_model, torch.nn.Module
                ), "dummy model must be provided when loading a model"

                # Assert that model architectures are the same
                valid = __validate_model(result, dummy_model)
                comm.get().broadcast(valid, src=src)  # Broadcast validation
                assert valid.item(), "Model architecture does not match loaded module"
                result.src = src
            # file contains unrecognized type
            else:
                # Broadcast load type
                load_type = torch.tensor(-1, dtype=torch.long)
                comm.get().broadcast(load_type, src=src)

                # raise error
                raise TypeError("Unrecognized load type %s" % type(result))

        # Non-source party
        else:
            # Receive load type from source party
            load_type = torch.tensor(-1, dtype=torch.long)
            comm.get().broadcast(load_type, src=src)

            # Load in tensor
            if load_type.item() == 0:
                # Receive size from source party
                dim = torch.empty(size=(), dtype=torch.long)
                comm.get().broadcast(dim, src=src)
                size = torch.empty(size=(dim.item(),), dtype=torch.long)
                comm.get().broadcast(size, src=src)
                result = cryptensor(torch.empty(size=tuple(size.tolist())), src=src)
            # Load module using dummy_model
            elif load_type.item() == 1:
                # Assert dummy_model is given
                assert dummy_model is not None and isinstance(
                    dummy_model, torch.nn.Module
                ), "dummy model must be provided when loading a model"
                result = dummy_model

                # Receive model architecture validation
                valid = torch.tensor(1, dtype=torch.long)
                comm.get().broadcast(valid, src=src)
                assert valid.item(), "Model architecture does not match loaded module"
                result.src = src
            else:
                raise TypeError("Unrecognized load type on src")
        # TODO: Encrypt modules before returning them
        return result


def save(obj, f, src=0, **kwargs):
    """
    Saves a CrypTensor or PyTorch tensor to a file.

    Args:
        obj: The CrypTensor or PyTorch tensor to be saved
        f: a file-like object (has to implement `read()`, `readline()`,
              `tell()`, and `seek()`), or a string containing a file name
        src: The source party that writes data to the specified file.
    """
    if is_encrypted_tensor(obj):
        raise NotImplementedError("Saving encrypted tensors is not yet supported")
    else:
        assert isinstance(src, int), "Save failed: src must be an integer"
        assert (
            src >= 0 and src < comm.get().get_world_size()
        ), "Save failed: src must be an integer in [0, world_size)"

        if comm.get().get_rank() == src:
            torch.save(obj, f, **kwargs)

    # Implement barrier to avoid race conditions that require file to exist
    comm.get().barrier()


def where(condition, input, other):
    """
    Return a tensor of elements selected from either `input` or `other`, depending
    on `condition`.
    """
    if is_encrypted_tensor(condition):
        return condition * input + (1 - condition) * other
    elif torch.is_tensor(condition):
        condition = condition.float()
    return input * condition + other * (1 - condition)


def cat(tensors, dim=0):
    assert isinstance(tensors, list), "input to cat must be a list"
    if len(tensors) == 1:
        return tensors[0]

    from .autograd_cryptensor import AutogradCrypTensor

    if any(isinstance(t, AutogradCrypTensor) for t in tensors):
        if not isinstance(tensors[0], AutogradCrypTensor):
            tensors[0] = AutogradCrypTensor(tensors[0], requires_grad=False)
        return tensors[0].cat(*tensors[1:], dim=dim)
    else:
        return get_default_backend().cat(tensors, dim=dim)


def stack(tensors, dim=0):
    assert isinstance(tensors, list), "input to stack must be a list"
    if len(tensors) == 1:
        return tensors[0].unsqueeze(dim)

    from .autograd_cryptensor import AutogradCrypTensor

    if any(isinstance(t, AutogradCrypTensor) for t in tensors):
        if not isinstance(tensors[0], AutogradCrypTensor):
            tensors[0] = AutogradCrypTensor(tensors[0], requires_grad=False)
        return tensors[0].stack(*tensors[1:], dim=dim)
    else:
        return get_default_backend().stack(tensors, dim=dim)


# Top level tensor functions
__PASSTHROUGH_FUNCTIONS = ["bernoulli", "rand", "randperm"]


def __add_top_level_function(func_name):
    def _passthrough_function(*args, backend=None, **kwargs):
        if backend is None:
            backend = get_default_backend()
        return getattr(backend, func_name)(*args, **kwargs)

    globals()[func_name] = _passthrough_function


for func in __PASSTHROUGH_FUNCTIONS:
    __add_top_level_function(func)

# expose classes and functions in package:
__all__ = ["CrypTensor", "debug", "init", "init_thread", "mpc", "nn", "uninit"]
