class AbstractSecretKey():

    def __init__(self, sk):
        return NotImplemented

    def decrypt(self, x):
        return NotImplemented

    def serialize(self):
        return NotImplemented


class AbstractPublicKey():

    def __init__(self, pk):
        return NotImplemented

    def encrypt(self, x):
        return NotImplemented

    def serialize(self):
        return NotImplemented


class AbstractKeyPair():

    def __init__(self):
        ""

    def deserialize(self, pubkey, seckey):
        return NotImplemented

    def generate(self, n_length=1024):
        return NotImplemented
