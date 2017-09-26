import syft


def Paillier(n_length=1024):
    return syft.he.paillier.keys.KeyPair().generate(n_length)
