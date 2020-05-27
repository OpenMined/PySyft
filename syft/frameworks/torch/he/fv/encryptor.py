from syft.frameworks.torch.he.fv.secret_key import SecretKey
from syft.frameworks.torch.he.fv.public_key import PublicKey
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_symmetric
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_asymmetric
from syft.frameworks.torch.he.fv.util.operations import multiply_add_plain_with_delta


class Encryptor:
    """Encrypts Plaintext objects into Ciphertext objects. Constructing an Encryptor
    requires a Context with valid encryption parameters, the public key or the secret
    key. If an Encrytor is given a secret key, it supports symmetric-key encryption.
    If an Encryptor is given a public key, it supports asymmetric-key encryption.

    Args:
        context (Context): Context for extracting encryption parameters.
        key: A public key or secret key that we want to use for encryption.
    """

    def __init__(self, context, key):
        self._context = context
        self._key = key

    def encrypt(self, message):
        """Encrypts an Plaintext data using the FV HE Scheme.

        Args:
            message (Plaintext): An plaintext which has to be encrypted.

        Retruns:
            A Ciphertext object containing the encrypted result.

        Raises:
            ValueError: Key provided for encryption is not a valid key object.
        """

        if isinstance(self._key, PublicKey):
            return self._encrypt(message, True)

        elif isinstance(self._key, SecretKey):
            return self._encrypt(message, False)

        else:
            raise ValueError("key for encryption is not valid")

    def _encrypt(self, message, is_asymmetric):
        """Encrypts the message according to the key provided.

        Args:
            message (Plaintext): An Plaintext object which has to be encrypted.
            is_asymmetric (bool): Based on the key provided for encryption select
                the mode of encryption.

        Returns:
            A Ciphertext object contating the encrypted result.
        """

        result = None
        if is_asymmetric:
            result = encrypt_asymmetric(
                self._context, self._key.data
            )  # Public key used for encryption

        else:
            result = encrypt_symmetric(
                self._context, self._key.data
            )  # Secret key used for encryption

        return multiply_add_plain_with_delta(result, message, self._context)
