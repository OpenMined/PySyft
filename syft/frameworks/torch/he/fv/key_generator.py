from syft.frameworks.torch.he.fv.context import Context
from syft.frameworks.torch.he.fv.util.rlwe import sample_poly_ternary
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_symmetric
from syft.frameworks.torch.he.fv.secret_key import SecretKey
from syft.frameworks.torch.he.fv.public_key import PublicKey
from syft.frameworks.torch.he.fv.relin_keys import RelinKeys


class KeyGenerator:
    """It is used for generating matching secret key and public key.
    Constructing a KeyGenerator requires only a Context class instance with valid
    encryption parameters.

    Args:
           context (Context): Context for extracting encryption parameters.
    """

    def __init__(self, context):
        if not isinstance(context, Context):
            raise ValueError("invalid context")

        self._public_key = None
        self._secret_key = None
        self._context = context
        self._relin_key_generator = None

    def keygen(self):
        """Generate the secret key and public key.

        Returns:
            A list of (size = 2) containing secret_key and public_key in respectively.
        """
        self._generate_sk()
        self._generate_pk()
        return [self._secret_key, self._public_key]

    def _generate_sk(self):
        param = self._context.param

        self._secret_key = SecretKey(sample_poly_ternary(param))

    def _generate_pk(self):
        if self._secret_key is None:
            raise RuntimeError("cannot generate public key for unspecified secret key")

        public_key = encrypt_symmetric(self._context, self._secret_key.data)
        self._public_key = PublicKey(public_key.data)

    def get_relin_keys(self):
        if self._relin_key_generator is None:
            if self._secret_key is None:
                raise RuntimeError("cannot generate relinearization key for unspecified secret key")

            self._relin_key_generator = RelinKeys(self._context, self._secret_key)

        # generate keys...
        return self._relin_key_generator._generate_relin_key()
