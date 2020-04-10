import torch as th

from syft.frameworks.torch.he.fv.context import Context
from syft.frameworks.torch.he.fv.util.rlwe import sample_poly_ternary
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_zero_symmetric
from syft.frameworks.torch.he.fv.plaintext import PlainText
from syft.frameworks.torch.he.fv.ciphertext import CipherText


class KeyGenerator:
    """Generates matching secret key and public key. An existing KeyGenerator can
    also at any time be used to generate relinearization keys and Galois keys.
    Constructing a KeyGenerator requires only a context object."""

    def __init__(self, context):
        """Creates a KeyGenerator initialized with the specified context params.

        Args:
            context: The Context object.
        """

        if not isinstance(context, Context):
            raise ValueError("invalid context")

        self._public_key = None
        self._secret_key = None
        self._context = context

        # Secret key and public key have not been generated
        self._sk_generated = False
        self._pk_generated = False

    def keygen(self):
        # Generate the secret and public key
        self.generate_sk()
        self.generate_pk()
        return [self._secret_key, self._public_key]

    def generate_sk(self, is_initialized=False):
        param = self._context.param

        if not is_initialized:
            self._secret_key = sample_poly_ternary(param)
            self._secret_key = PlainText(self._secret_key)
        self._sk_generated = True

    def generate_pk(self):
        if not self._sk_generated:
            raise RuntimeError("cannot generate public key for unspecified secret key")

        self._public_key = encrypt_zero_symmetric(self._context, self._secret_key)
        self._public_key = CipherText(self._public_key)
