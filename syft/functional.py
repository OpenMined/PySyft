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