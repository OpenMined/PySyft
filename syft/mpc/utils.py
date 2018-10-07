def egcd(a, b):
    """
    greatest common denominator
    :param a:
    :param b:
    :return:
    """

    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    """
    calculate the multiplicative inverse of a modulus m such that

    (x * result) % m == (x / a)

    for any integer between 0 and m

    :param a: the number we wish to divide by
    :param m: the size of the modular field
    :return: the number we can multiply by to actually divide by a
    """

    g, x, y = egcd(a, m)
    if g != 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m