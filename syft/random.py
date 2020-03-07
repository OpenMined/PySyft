from random import SystemRandom
cryptogen = SystemRandom()

def generate_random_id():
    return cryptogen.randrange(0, 2 ** 63 - 1)