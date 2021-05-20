# third party
from sympy.core.numbers import Rational

# syft relative
from .number_expression import generate_number_expression_type

generate_number_expression_type(real_type=Rational)
