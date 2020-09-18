from syft.frameworks.torch.he.fv.context import Context
from syft.frameworks.torch.he.fv.util.rlwe import sample_poly_ternary
from syft.frameworks.torch.he.fv.util.rlwe import encrypt_symmetric
from syft.frameworks.torch.he.fv.secret_key import SecretKey
from syft.frameworks.torch.he.fv.public_key import PublicKey
from syft.frameworks.torch.he.fv.relin_keys import RelinKeys


class KeyGenerator:
    """Used to generates secret, public key and relinearization key.

    Args:
           context (Context): Context for extracting encryption parameters.
    """

    def __init__(self, context):
        if not isinstance(context, Context):
            raise RuntimeError("invalid context")

        self.public_key = None
        self.secret_key = None
        self.context = context
        self.relin_key_generator = None

    def keygen(self):
        """Generate the secret key and public key.

        Returns:
            A list of (size = 2) containing secret_key and public_key in respectively.
        """
        self._generate_sk()
        self._generate_pk()
        return [self.secret_key, self.public_key]

    def _generate_sk(self):
        param = self.context.context_data_map[self.context.key_param_id].param
        self.secret_key = SecretKey(sample_poly_ternary(param))

    def _generate_pk(self):
        if self.secret_key is None:
            raise RuntimeError("cannot generate public key for unspecified secret key")

        public_key = encrypt_symmetric(
            self.context, self.context.key_param_id, self.secret_key.data
        )
        self.public_key = PublicKey(public_key.data)

    def get_relin_keys(self):
        """Generate a relinearization key.

        Returns:
            A relinearization key.
        """
        if self.relin_key_generator is None:
            if self.secret_key is None:
                raise RuntimeError("cannot generate relinearization key for unspecified secret key")

            self.relin_key_generator = RelinKeys(self.context, self.secret_key)

        # generate keys.
        return self.relin_key_generator._generate_relin_key()
