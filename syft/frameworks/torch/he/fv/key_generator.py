from syft.frameworks.torch.he.fv.context import Context
from syft.frameworks.torch.he.fv.util.rlwe import sample_poly_ternary


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
        self._secret_key_size = 0

        # Secret key and public key have not been generated
        self._sk_generated = False
        self._pk_generated = False

        # Generate the secret and public key
        self.generate_sk()
        self.generate_pk()

    def generate_sk(self, is_initialized=False):
        param = self._context.param
        coeff_modulus = param.coeff_modulus
        coeff_count = param.poly_modulus_degree
        coeff_mod_count = len(coeff_modulus)

        if not is_initialized:
            self._sk_generated = False
            self.secret_key = sample_poly_ternary(param)
        self._sk_generated = True
        return self.secret_key

    def generate_pk(self):
        if not self._sk_generated:
            raise RuntimeError("cannot generate public key for unspecified secret key")
        pass
