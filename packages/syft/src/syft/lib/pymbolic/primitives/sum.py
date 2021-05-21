# third party
from pymbolic.primitives import Sum

# syft relative
from .multi_child_expression import generate_multi_child_expression_type

generate_multi_child_expression_type(real_type=Sum)
