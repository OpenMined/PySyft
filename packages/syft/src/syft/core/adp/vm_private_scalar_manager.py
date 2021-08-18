# CLEANUP NOTES:
# - remove unused comments
# - add documentation for each method
# - add comments inline explaining each piece
# - add a unit test for each method (at least)
# - Sort out joins between PrimeFactory objects
# - (after previous bullet is done) every tensor has its own prime factory

# third party
import sympy as sp

# relative
from ..common.serde.recursive import RecursiveSerde
from .entity import Entity
from .scalar import GammaScalar
from .types import AcceptableSimpleType


class PrimeFactory(RecursiveSerde):

    """IMPORTANT: it's very important that two tensors be able to tell that
    they are indeed referencing the EXACT same PrimeFactory. At present this is done
    by ensuring that it is literally the same python object. In the future, we will probaby
    need to formalize this. However, the main way this could go wrong is if we created some
    alternate way for checking to see if two prime factories 'sortof looked the same' but which
    in fact weren't the EXACT same object. This could lead to security leaks wherein two tensors
    think two different symbols in fact are the same symbol."""

    __attr_allowlist__ = ["prev_prime"]

    def __init__(self) -> None:
        self.prev_prime = 1

    def next(self) -> int:
        self.prev_prime = sp.nextprime(self.prev_prime)
        return self.prev_prime


class VirtualMachinePrivateScalarManager(RecursiveSerde):

    __attr_allowlist__ = ["prime_factory", "prime2symbol"]

    def __init__(self) -> None:
        self.prime_factory = PrimeFactory()
        self.prime2symbol = {}

    def get_symbol(
        self,
        min_val: AcceptableSimpleType,
        value: AcceptableSimpleType,
        max_val: AcceptableSimpleType,
        entity: Entity,
    ) -> int:
        gs = GammaScalar(min_val=min_val, value=value, max_val=max_val, entity=entity)
        gs.prime = self.prime_factory.next()
        self.prime2symbol[gs.prime] = gs
        return gs.prime
