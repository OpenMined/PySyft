import random
from syft.frameworks.torch.he.fv.util.operations import exponentiate_mod
from syft.frameworks.torch.he.fv.util.operations import multiply_mod


def is_prime(value, num_rounds=40):
    """Check for the integer if it is probably a prime number.
    Not intrested in strictly checking for prime.

    Returns:
        True, if it is probably prime number else False.
    """
    # First check the simplest cases.
    if value < 2:
        return False
    if value == 2:
        return True
    if 0 == value % 2:
        return False
    if 3 == value:
        return True
    if 0 == value % 3:
        return False
    if 5 == value:
        return True
    if 0 == value % 5:
        return False
    if 7 == value:
        return True
    if 0 == value % 7:
        return False
    if 11 == value:
        return True
    if 0 == value % 11:
        return False
    if 13 == value:
        return True
    if 0 == value % 13:
        return False

    # Second, Miller-Rabin test.
    # Find r and odd d that satisfy value = 2^r * d + 1.
    d = value - 1
    r = 0
    while 0 == d & 1:
        d >>= 1
        r += 1

    if r == 0:
        return False

    # 1) Pick a = 2, check a^(value - 1).
    # 2) Pick a randomly from [3, value - 1], check a^(value - 1).
    # 3) Repeat 2) for another num_rounds - 1 times.
    for i in range(num_rounds):
        a = random.randint(3, value - 1) if i != 0 else 2
        x = exponentiate_mod(a, d, value)

        if x == 1 or x == value - 1:
            continue
        count = 0

        while True:
            x = multiply_mod(x, x, value)
            count += 1
            if not (x != value - 1 and count < r - 1):
                break

        if x != value - 1:
            return False
    return True


def get_primes(size, bit_size, count):
    """Generate a list of probably prime numbers with at least a constant
    factor difference between the numbers.

    Args:
        size: polynomail modulus value.
        bit_size: bit lengths of the prime no's to be generated.
        count: No's of prime numbers required.

    Returns:
        A list of probably primes numbers.

    Raises:
        ValueError: if `count <= 0` or `size <= 0`.
    """
    if count <= 0:
        raise ValueError(f"{count} must be positive value.")
    if size <= 0:
        raise ValueError(f"{size} must be positive value.")

    result = []
    factor = 2 * size

    # Start with 2^bit_size - 2 * size + 1
    value = 1 << bit_size
    value = value - factor + 1

    lower_bound = 1 << (bit_size - 1)
    while count > 0 and value > lower_bound:
        if is_prime(value):
            result.append(value)
            count -= 1
        value -= factor
    if count > 0:
        raise RuntimeError("failed to find enough qualifying primes")

    return result
