import torch as th

from syft.frameworks.torch.he.fv.ciphertext import CipherText
from syft.frameworks.torch.he.fv.secret_key import SecretKey
from syft.frameworks.torch.he.fv.public_key import PublicKey
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_zero_symmetric
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_zero_asymmetric
from syft.frameworks.torch.he.fv.util.operations import multiply_add_plain_with_scaling_variant


class Encrypter:
    """Encrypts Plaintext objects into Ciphertext objects. Constructing an Encryptor
    requires a Context with valid encryption parameters, the public key or
    the secret key. If an Encrytor is given a secret key, it supports symmetric-key
    encryption. If an Encryptor is given a public key, it supports asymmetric-key
    encryption."""

    def __init__(self, context, key):
        self._context = context
        self._key = key

    def encrypt_internal(self, message, is_asymmetric):
        result = CipherText([])
        if is_asymmetric:
            result = encrypt_zero_asymmetric(self._context, self._key)
        else:
            result = encrypt_zero_symmetric(self._context, self._key)

        return multiply_add_plain_with_scaling_variant(result, message, self._context)

    def encrypt(self, message):
        if isinstance(self._key, SecretKey):
            # Secret key used for encryption
            return self.encrypt_internal(message, False)
        elif isinstance(self._key, PublicKey):
            # Public key used for encryption
            return self.encrypt_internal(message, True)
        else:
            raise ValueError("key for encryption is not valid")
