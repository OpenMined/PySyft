"""
This module contains the functions to be benchmarked.
"""

import torch


def sigmoid(method: str, prec_frac: int, workers: dict):
    """
    Function to simulate a sigmoid approximation, given
    a method, a precision value and the workers used
    for sharing data.
    """

    # Define workers
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    # Init tensor, share and approximate sigmoid
    example_tensor = torch.tensor([1.23212])
    t_sh = example_tensor.fix_precision(precision_fractional=prec_frac).share(
        alice, bob, crypto_provider=james
    )
    r_sh = t_sh.sigmoid(method=method)
    return r_sh.get().float_prec()


def tanh(method: str, prec_frac: int, workers: dict):
    """
    Function to simulate a tanh approximation, given
    a method, a precision value and the workers used
    for sharing data.
    """

    # Define workers
    alice, bob, james = workers["alice"], workers["bob"], workers["james"]

    # Init tensor, share and approximate sigmoid
    example_tensor = torch.tensor([1.23212])
    t_sh = example_tensor.fix_precision(precision_fractional=prec_frac).share(
        alice, bob, crypto_provider=james
    )
    r_sh = t_sh.tanh(method=method)
    return r_sh.get().float_prec()
