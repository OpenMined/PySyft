# stdlib
from os import urandom
import secrets
import string


def generate_sec_random_password(length):
    if not isinstance(length, int) or length < 10:
        raise ValueError(
            "Password should have a positive safe length of at least 10 characters!"
        )

    alphabet = string.ascii_letters + string.digits + string.punctuation

    # original Python 2 (urandom returns str)
    # return "".join(chars[ord(c) % len(chars)] for c in urandom(length))

    # Python 3 (urandom returns bytes)
    return "".join(alphabet[c % len(alphabet)] for c in urandom(length))
