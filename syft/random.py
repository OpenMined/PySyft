from random import SystemRandom

cryptogen = SystemRandom()

# The size of all IDs which PySyft generates
SYFT_ID_SIZE = 2 ** 63 - 1


def generate_cryptographically_secure_random_id():
    """Generate a random ID using a cryptographically secure technique.

    For all objects in the PySyft library, we need to generate IDs. All ID generation
    uses this method.
    """

    return cryptogen.randrange(0, SYFT_ID_SIZE)
