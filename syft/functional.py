from syft.controller import cmd, params_func, no_params_func

def concatenate(tensors,axis=0):
    ids = list()
    for t in tensors:
        ids.append(t.id)
    return params_func(cmd, "concatenate", params=[axis] + ids, return_type='FloatTensor')

def ones(*args):
    """
    Returns a tensor filled with zeros
    The shape of the tensor is defined by the varargs sizes.
    ----------
    Returns
    -------
    FloatTensor
    """
    dims = list(args)
    assert type(dims[0]) == int
    return params_func(cmd, "ones", params=dims, return_type='FloatTensor')

def randn(*args):
    """
    Returns a tensor filled with random numbers from a normal distribution with mean=0 and std=1
    The shape of the tensor is defined by the varargs sizes.
    ----------
    Returns
    -------
    FloatTensor
    """
    dims = list(args)
    assert type(dims[0]) == int
    return params_func(cmd, "randn", params=dims, return_type='FloatTensor')

def random(*args):
    """
    Returns a tensor filled with random numbers from a uniform distribution on the interval [0,1)
    The shape of the tensor is defined by the varargs sizes.
    ----------
    Returns
    -------
    FloatTensor
    """
    dims = list(args)
    assert type(dims[0]) == int
    return params_func(cmd, "random", params=dims, return_type='FloatTensor')

def set_seed(seed):
    """
    Sets the seed value for the random number generator to make model testing deterministic. 
    """
    assert (type(seed) == int and seed >= 0)
    return params_func(cmd, "set_seed", params=[seed])

def zeros(*args):
    """
    Returns a tensor filled with zeros
    The shape of the tensor is defined by the varargs sizes.
    ----------
    Returns
    -------
    FloatTensor
    """
    dims = list(args)
    assert type(dims[0]) == int
    return params_func(cmd, "zeros", params=dims, return_type='FloatTensor')