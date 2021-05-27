import sympy as sp
import uuid
import numpy as np
from collections import defaultdict
from .intermediate_gamma import IntermediateGammaTensor

class PrimeFactory():

    def __init__(self):
        self.prev_prime = 1
        self.placement2prime = defaultdict(dict)
        self.prime2placement = defaultdict(dict)

    def next(self, uid, i):
        self.prev_prime = sp.nextprime(self.prev_prime)
        self.placement2prime[uid][i] = self.prev_prime
        self.prime2placement[self.prev_prime][i] = uid
        return self.prev_prime


class InitialGammaTensor(IntermediateGammaTensor):

    def __init__(self, values, min_vals, max_vals, entities, symbol_factory=None):
        self.uid = uuid.uuid4()
        self.values = values  # child
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.entities = entities

        if symbol_factory == None:
            symbol_factory = PrimeFactory()

        self.symbol_factory = symbol_factory

        some_symbols = list()
        for i in range(self.values.flatten().shape[0]):
            some_symbols.append(symbol_factory.next(self.uid, i))

        term_tensor = np.array(some_symbols).reshape(list(self.values.shape) + [1])
        coeff_tensor = (term_tensor * 0) + 1
        bias_tensor = (self.values * 0)

        super().__init__(term_tensor=term_tensor,
                         coeff_tensor=coeff_tensor,
                         bias_tensor=bias_tensor,
                         symbol_factory=self.symbol_factory)
