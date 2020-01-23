import numpy as np

from syft.frameworks.torch.tensors.interpreters.polynomial.term import Term
from syft.frameworks.torch.tensors.interpreters.polynomial.variable import Variable
from syft.frameworks.torch.tensors.interpreters.polynomial.polynomial import Polynomial
from syft.frameworks.torch.tensors.interpreters.polynomial.tensor import PolynomialTensor


def var(name, minimum_factor=0):
    return Polynomial(Term(variables=Variable(name)), minimum_factor=minimum_factor)


def var_tensor(shape, minimum_factor=0.1):
    size = 1
    for d in shape:
        size *= d

    v = list()
    for i in range(size):
        v.append(var("x_" + str(i), minimum_factor=minimum_factor))

    vartensor = PolynomialTensor(np.array(v)).wrap()
    return vartensor.view(*shape)


def get_factor_vector(result):
    factors = list()
    for poly in result.flatten():

        for term in poly.additive_terms:
            factors.append(term.factor)

        factors.append(1)

    factors = np.array(factors)
    return factors


def get_input_kwargs(input_vector):
    var2value = {}

    for i, v in enumerate(input_vector.view(-1)):
        var2value["x_" + str(i)] = float(v)

    return var2value


def get_expanded_input_vector(input_vector, poly_pred_array):

    var2value = get_input_kwargs(input_vector)

    results = list()
    for poly_pred in poly_pred_array.flatten():

        expanded_input_vector = list()
        for term in poly_pred.additive_terms:
            expanded_input_vector.append(term(ignore_factor=True, **var2value))

        expanded_input_vector.append(poly_pred.additive_constant)

        expanded_input_vector = np.array(expanded_input_vector)
        results.append(expanded_input_vector)

    return np.array(results)
