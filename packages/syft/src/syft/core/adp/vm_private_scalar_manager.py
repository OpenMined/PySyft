# third party
import sympy as sp

# syft relative
from .scalar import GammaScalar


class PrimeFactory:

    """IMPORTANT: it's very important that two tensors be able to tell that
    they are indeed referencing the EXACT same PrimeFactory. At present this is done
    by ensuring that it is literally the same python object. In the future, we will probaby
    need to formalize this. However, the main way this could go wrong is if we created some
    alternate way for checking to see if two prime factories 'sortof looked the same' but which
    in fact weren't the EXACT same object. This could lead to security leaks wherein two tensors
    think two different symbols in fact are the same symbol."""

    def __init__(self):
        self.prev_prime = 1

    def next(self):
        self.prev_prime = sp.nextprime(self.prev_prime)
        return self.prev_prime


class VirtualMachinePrivateScalarManager:
    def __init__(self):
        self.prime_factory = PrimeFactory()
        self.prime2symbol = {}

    def get_symbol(self, min_val, value, max_val, entity):
        gs = GammaScalar(min_val=min_val, value=value, max_val=max_val, entity=entity)
        gs.prime = self.prime_factory.next()
        self.prime2symbol[gs.prime] = gs
        return gs.prime
