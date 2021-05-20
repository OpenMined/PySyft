# third party
from sympy.core.mul import Mul

# syft relative
from .multi_child_expression import generate_children_type

generate_children_type(real_type=Mul)
