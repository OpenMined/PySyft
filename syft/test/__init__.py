from syft.generic.utils import allow_command


@allow_command
def my_awesome_computation(x):
    x = x + 1
    x = x * 2
    x = x - 1
    y = x * 2
    return x, y
