# third party
from sympy.core.mul import Mul

# syft relative
from .args_expression import generate_args_expression_type

generate_args_expression_type(real_type=Mul)
