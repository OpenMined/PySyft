# stdlib
from os import urandom
import string
import sys
from typing import List
from typing import Set


def generate_sec_random_password(
    length: int,
    special_chars: bool = True,
    digits: bool = True,
    lower_case: bool = True,
    upper_case: bool = True,
) -> str:
    """Generates a random password of the given length.

    Args:
        length (int): length of the password
        special_chars (bool, optional): Include at least one specials char in the password. Defaults to True.
        digits (bool, optional): Include at least one digit in the password. Defaults to True.
        lower_case (bool, optional): Include at least one lower case character in the password. Defaults to True.
        upper_case (bool, optional): Includde at least one upper case character in the password. Defaults to True.

    Raises:
        ValueError: If password length if too short.

    Returns:
        str: randomly generated password
    """
    if not isinstance(length, int) or length < 10:
        raise ValueError(
            "Password should have a positive safe length of at least 10 characters!"
        )

    choices: str = ""
    required_tokens: List[str] = []
    if special_chars:
        special_characters = "!@#$%^&*()_+"
        choices += special_characters
        required_tokens.append(
            special_characters[
                int.from_bytes(urandom(1), sys.byteorder) % len(special_characters)
            ]
        )
    if lower_case:
        choices += string.ascii_lowercase
        required_tokens.append(
            string.ascii_lowercase[
                int.from_bytes(urandom(1), sys.byteorder) % len(string.ascii_lowercase)
            ]
        )
    if upper_case:
        choices += string.ascii_uppercase
        required_tokens.append(
            string.ascii_uppercase[
                int.from_bytes(urandom(1), sys.byteorder) % len(string.ascii_uppercase)
            ]
        )
    if digits:
        choices += string.digits
        required_tokens.append(
            string.digits[
                int.from_bytes(urandom(1), sys.byteorder) % len(string.digits)
            ]
        )

    # Python 3 (urandom returns bytes)
    password = [choices[c % len(choices)] for c in urandom(length)]

    # Pick some random indexes
    random_indexes: Set[int] = set()
    while len(random_indexes) < len(required_tokens):
        random_indexes.add(int.from_bytes(urandom(1), sys.byteorder) % len(password))

    # Replace the random indexes with the required tokens
    for i, idx in enumerate(random_indexes):
        password[idx] = required_tokens[i]

    return "".join(password)


if __name__ == "__main__":
    pwd_length = 48
    # generate_sec_random_password(pwd_length)
    print(generate_sec_random_password(pwd_length, special_chars=False))
