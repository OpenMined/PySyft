# This script reads in derivative definitions from derivatives.yaml and writes appropriately
# constructed GradFunc classes to gradients.py. It could probably be optimized.

import re
import yaml

tab = " " * 4


def split_signature(signature):
    return re.match(r"^(\S+)\((.+)\)", signature).groups()


# There is probably a better way to do this
# Need to handle keyword arguments as well
def construct_grad_fn_class(grad_def):
    lines = []

    name, arguments = split_signature(grad_def["name"])

    # This won't work right if keyword arguments are present. I should refactor
    # this with it's own function. TODO
    input_args = arguments.split(", ")
    signature = arguments.replace("self", "self_")

    lines.append(f"class {name.capitalize()}Backward(GradFunc):")
    lines.append(tab + f"def __init__(self, {signature}):")
    lines.append(2 * tab + f"super().__init__(self, {signature})")
    for arg in signature.split(", "):
        lines.append(2 * tab + f"self.{arg} = {arg}")
    lines.append("")

    lines.append(tab + "def gradient(self, grad):")
    for arg in input_args:
        formula = grad_def[arg]
        for in_arg in input_args:
            formula = formula.replace(in_arg, f"self.{in_arg}")
        formula = formula.replace(".self", ".self_").replace("result", "self.result")
        lines.append(2 * tab + f"grad_{arg.replace('self', 'self_')} = {formula}")
    grad_signature = ", ".join([f"grad_{arg}".replace("self", "self_") for arg in input_args])
    lines.append(2 * tab + f"return ({grad_signature},)")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    with open("syft/frameworks/torch/tensors/interpreters/derivatives.yaml", "r") as f:
        derivatives = yaml.safe_load(f.read())

    with open("syft/frameworks/torch/tensors/interpreters/gradients.py", "w") as f:
        grad_fn_map = {}
        f.write("# This file is generated from build_gradients.py\n\n")
        f.write("from . gradients_core import *\n\n")
        for definition in derivatives:
            f.write(construct_grad_fn_class(definition))
            f.write("\n")
