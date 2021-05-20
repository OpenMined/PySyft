# third party
from sympy.core.add import Add

# syft relative
from .args_expression import generate_args_expression_type

generate_args_expression_type(real_type=Add)
