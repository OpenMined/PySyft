import copy

from syft.frameworks.torch.he.fv.util.operations import poly_add_mod
from syft.frameworks.torch.he.fv.util.operations import multiply_add_plain_with_delta
from syft.frameworks.torch.he.fv.ciphertext import CipherText
from syft.frameworks.torch.he.fv.plaintext import PlainText
from syft.frameworks.torch.he.fv.integer_encoder import IntegerEncoder


class Evaluator:
    def __init__(self, context):
        self.context = context
        self.coeff_modulus = context.param.coeff_modulus

    def add(self, op1, op2):
        """Adds two operands using FV scheme.

        Args:
            op1 (Ciphertext/Plaintext): First argument.
            op2 (Ciphertext/Plaintext): Second argument.

        Returns:
            If both arguments are Plaintext elements then the result will be a Plaintext object
                otherwise a Ciphertext object with value equivalent to the result of addition
                operation of two provided arguments.
        """
        if isinstance(op1, CipherText) and isinstance(op2, CipherText):
            return self._add_cipher_cipher(op1, op2)

        elif isinstance(op1, PlainText) and isinstance(op2, PlainText):
            return self._add_plain_plain(op1, op2)

        elif isinstance(op1, PlainText) and isinstance(op2, CipherText):
            return self._add_plain_cipher(op1, op2)

        elif isinstance(op1, CipherText) and isinstance(op2, PlainText):
            return self._add_plain_cipher(op2, op1)

        else:
            raise TypeError(f"Addition Operation not supported between {type(op1)} and {type(op2)}")

    def _add_cipher_cipher(self, ct1, ct2):
        """Adds two ciphertexts.

        Args:
            ct1 (Ciphertext): First argument.
            ct2 (Ciphertext): Second argument.

        Returns:
            A Ciphertext object with value equivalent to result of addition of two provided
                arguments.
        """
        ct1, ct2 = copy.deepcopy(ct1.data), copy.deepcopy(ct2.data)
        result = ct2 if len(ct2) > len(ct1) else ct1

        for i in range(min(len(ct1), len(ct2))):
            for j in range(len(self.coeff_modulus)):
                result[i][j] = poly_add_mod(ct1[i][j], ct2[i][j], self.coeff_modulus[j])

        return CipherText(result)

    def _add_plain_cipher(self, pt, ct):
        """Adds a ciphertext and a plaintext.

        Args:
            pt (Plaintext): First argument.
            ct (Ciphertext): Second argument.
        Returns:
            A Ciphertext object with value equivalent to result of addition of two provided
                arguments.
        """
        ct = copy.deepcopy(ct)
        return multiply_add_plain_with_delta(ct, pt, self.context)

    def _add_plain_plain(self, pt1, pt2):
        """Adds two plaintexts object.

        Args:
            pt1 (Plaintext): First argument.
            pt2 (Plaintext): Second argument.

        Returns:
            A Plaintext object with value equivalent to result of addition of two provided
                arguments.
        """
        encoder = IntegerEncoder(self.context)

        return encoder.encode(encoder.decode(pt1) + encoder.decode(pt2))
