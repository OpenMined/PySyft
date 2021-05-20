# third party
from sympy.core.numbers import Integer

# syft relative
from .number_expression import generate_number_expression_type

generate_number_expression_type(real_type=Integer)
