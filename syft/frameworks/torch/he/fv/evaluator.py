import copy
from enum import Enum

from syft.frameworks.torch.he.fv.util.operations import poly_add
from syft.frameworks.torch.he.fv.util.operations import poly_add_mod
from syft.frameworks.torch.he.fv.util.operations import poly_mul
from syft.frameworks.torch.he.fv.util.operations import poly_mul_mod
from syft.frameworks.torch.he.fv.util.operations import negate_mod
from syft.frameworks.torch.he.fv.util.operations import poly_sub_mod
from syft.frameworks.torch.he.fv.util.operations import poly_negate_mod
from syft.frameworks.torch.he.fv.util.operations import poly_mul_mod
from syft.frameworks.torch.he.fv.util.operations import multiply_add_plain_with_delta
from syft.frameworks.torch.he.fv.util.operations import multiply_sub_plain_with_delta
from syft.frameworks.torch.he.fv.ciphertext import CipherText
from syft.frameworks.torch.he.fv.plaintext import PlainText


class ParamTypes(Enum):
    """Enumeration for type checking of parameters."""

    CTCT = 1
    PTPT = 2
    CTPT = 3
    PTCT = 4


def _typecheck(op1, op2):
    """Check the type of parameters used and return correct enum type."""
    if isinstance(op1, CipherText) and isinstance(op2, CipherText):
        return ParamTypes.CTCT
    elif isinstance(op1, PlainText) and isinstance(op2, PlainText):
        return ParamTypes.PTPT
    elif isinstance(op1, CipherText) and isinstance(op2, PlainText):
        return ParamTypes.CTPT
    elif isinstance(op1, PlainText) and isinstance(op2, CipherText):
        return ParamTypes.PTCT
    else:
        return None


class Evaluator:
    def __init__(self, context):
        self.context = context
        self.poly_modulus = context.param.poly_modulus
        self.coeff_modulus = context.param.coeff_modulus
        self.plain_modulus = context.param.plain_modulus

    def add(self, op1, op2):
        """Add two operands using FV scheme.

        Args:
            op1 (Ciphertext/Plaintext): First polynomial argument (Augend).
            op2 (Ciphertext/Plaintext): Second polynomial argument (Addend).

        Returns:
            If both arguments are Plaintext elements then the result will be a Plaintext object
                otherwise a Ciphertext object with value equivalent to the result of addition
                operation of two provided arguments.
        """

        param_type = _typecheck(op1, op2)

        if param_type == ParamTypes.CTCT:
            return self._add_cipher_cipher(op1, op2)

        elif param_type == ParamTypes.PTPT:
            return self._add_plain_plain(op1, op2)

        elif param_type == ParamTypes.CTPT:
            return self._add_cipher_plain(op1, op2)

        elif param_type == ParamTypes.PTCT:
            return self._add_cipher_plain(op2, op1)

        else:
            raise TypeError(f"Addition Operation not supported between {type(op1)} and {type(op2)}")

    def sub(self, op1, op2):
        """Subtracts two operands using FV scheme.

        Args:
            op1 (Ciphertext/Plaintext): First polynomial argument (Minuend).
            op2 (Ciphertext/Plaintext): Second polynomial argument (Subtrahend).

        Returns:
            A ciphertext object with the value equivalent to the result of the subtraction
                of two operands.
        """
        param_type = _typecheck(op1, op2)

        if param_type == ParamTypes.CTCT:
            return self._sub_cipher_cipher(op1, op2)

        elif param_type == ParamTypes.CTPT:
            return self._sub_cipher_plain(op1, op2)

        elif param_type == ParamTypes.PTCT:
            return self._sub_cipher_plain(op2, op1)

        else:
            raise TypeError(
                f"Subtraction Operation not supported between {type(op1)} and {type(op2)}"
            )

    def negate(self, ct):
        """Negate a cipher i.e -(ct_value)

        Args:
            ct (Ciphertext): Ciphertext to be negated.

        Returns:
            A Ciphertext object with value equivalent to result of -(ct_value).
        """
        result = copy.deepcopy(ct.data)

        for i in range(len(result)):
            for j in range(len(result[i])):
                for k in range(len(result[i][j])):
                    result[i][j][k] = negate_mod(ct.data[i][j][k], self.coeff_modulus[j])

        return CipherText(result)

    def mul(self, op1, op2):
        """Multiply two operands using FV scheme.

        Args:
            op1 (Ciphertext/Plaintext): First polynomial argument (Multiplicand).
            op2 (Ciphertext/Plaintext): Second polynomial argument (Multiplier).

        Returns:
            A Ciphertext object with a value equivalent to the result of the product of two
                operands.
        """
        param_type = _typecheck(op1, op2)

        if param_type == ParamTypes.CTCT:
            return self._mul_cipher_cipher(op1, op2)

        elif param_type == ParamTypes.PTPT:
            return self._mul_plain_plain(op1, op2)

        elif param_type == ParamTypes.CTPT:
            return self._mul_cipher_plain(op1, op2)

        elif param_type == ParamTypes.PTCT:
            return self._mul_cipher_plain(op2, op1)

        else:
            raise TypeError(
                f"Multiplication Operation not supported between {type(op1)} and {type(op2)}"
            )

    def _add_cipher_cipher(self, ct1, ct2):
        """Adds two ciphertexts.

        Args:
            ct1 (Ciphertext): First polynomial argument (Augend).
            ct2 (Ciphertext): Second polynomial argument (Addend).

        Returns:
            A Ciphertext object with value equivalent to result of addition of two provided
                arguments.
        """
        ct1, ct2 = copy.deepcopy(ct1.data), copy.deepcopy(ct2.data)
        result = ct2 if len(ct2) > len(ct1) else ct1

        for i in range(min(len(ct1), len(ct2))):
            for j in range(len(self.coeff_modulus)):
                result[i][j] = poly_add_mod(
                    ct1[i][j], ct2[i][j], self.coeff_modulus[j], self.poly_modulus
                )

        return CipherText(result)

    def _add_cipher_plain(self, ct, pt):
        """Add a plaintext into a ciphertext.

        Args:
            ct (Ciphertext): First polynomial argument (Augend).
            pt (Plaintext): Second polynomial argument (Addend).

        Returns:
            A Ciphertext object with value equivalent to result of addition of two provided
                arguments.
        """
        ct = copy.deepcopy(ct)
        return multiply_add_plain_with_delta(ct, pt, self.context)

    def _add_plain_plain(self, pt1, pt2):
        """Adds two plaintexts object.

        Args:
            pt1 (Plaintext): First polynomial argument (Augend).
            pt2 (Plaintext): Second polynomial argument (Addend).

        Returns:
            A Plaintext object with value equivalent to result of addition of two provided
                arguments.
        """
        pt1, pt2 = copy.deepcopy(pt1), copy.deepcopy(pt2)
        return PlainText(poly_add_mod(pt1.data, pt2.data, self.plain_modulus, self.poly_modulus))

    def _sub_cipher_plain(self, ct, pt):
        """Subtract a plaintext from a ciphertext.

        Args:
            ct (Ciphertext): First polynomial argument (Minuend).
            pt (Plaintext): Second polynomial argument (Subtrahend).

        Returns:
            A Ciphertext object with value equivalent to result of addition of two provided
                arguments.
        """
        ct = copy.deepcopy(ct)
        return multiply_sub_plain_with_delta(ct, pt, self.context)

    def _sub_cipher_cipher(self, ct1, ct2):
        """Subtract two ciphertexts.

        Args:
            ct1 (Ciphertext): First polynomial argument (Minuend).
            ct2 (Ciphertext): Second polynomial argument (Subtrahend).

        Returns:
            A Ciphertext object with value equivalent to result of subtraction of two provided
                arguments.
        """
        ct1, ct2 = copy.deepcopy(ct1.data), copy.deepcopy(ct2.data)
        result = ct2 if len(ct2) > len(ct1) else ct1
        min_size, max_size = min(len(ct1), len(ct2)), max(len(ct1), len(ct2))

        for i in range(min_size):
            for j in range(len(self.coeff_modulus)):
                result[i][j] = poly_sub_mod(
                    ct1[i][j], ct2[i][j], self.coeff_modulus[j], self.poly_modulus
                )

        for i in range(min_size + 1, max_size):
            for j in range(len(self.coeff_modulus)):
                result[i][j] = poly_negate_mod(result[i][j], self.coeff_modulus[j])

        return CipherText(result)

    def _mul_cipher_cipher(self, ct1, ct2):
        """Multiply two operands using FV scheme.

        Args:
            op1 (Ciphertext): First polynomial argument (Multiplicand).
            op2 (Ciphertext): Second polynomial argument (Multiplier).

        Returns:
            A Ciphertext object with a value equivalent to the result of the product of two
                operands.
        """
        ct1, ct2 = ct1.data, ct2.data

        if len(ct1) > 2:
            # TODO: perform relinearization operation.
            raise RuntimeError(
                "Cannot multiply ciphertext of size >2, Perform relinearization operation."
            )
        if len(ct2) > 2:
            # TODO: perform relinearization operation.
            raise RuntimeError(
                "Cannot multiply ciphertext of size >2, Perform relinearization operation."
            )

        # Now the size of ciphertexts is 2.
        # Multiplication operation of ciphertext:
        #   result = [r1, r2, r3] where
        #   r1 = ct1[0] * ct2[0]
        #   r2 = ct1[0] * ct2[1] + ct1[1] * ct2[0]
        #   r3 = ct1[1] * ct2[1]
        #
        # where ct1[i], ct2[i] are polynomials.

        ct10, ct11 = ct1
        ct20, ct21 = ct2

        result = [
            [0] * len(self.coeff_modulus),
            [0] * len(self.coeff_modulus),
            [0] * len(self.coeff_modulus),
        ]

        for i in range(len(self.coeff_modulus)):
            result[0][i] = poly_mul(ct10[i], ct20[i], self.poly_modulus)

            result[1][i] = poly_add(
                poly_mul(ct11[i], ct20[i], self.poly_modulus),
                poly_mul(ct10[i], ct21[i], self.poly_modulus),
                self.poly_modulus,
            )

            result[2][i] = poly_mul(ct11[i], ct21[i], self.poly_modulus)

        for i in range(len(result)):
            for j in range(len(self.coeff_modulus)):
                result[i][j] = [
                    round(((x * self.plain_modulus) / self.coeff_modulus[j]))
                    % self.coeff_modulus[j]
                    for x in result[i][j]
                ]

        return CipherText(result)

    def _mul_cipher_plain(self, ct, pt):
        """Multiply two operands using FV scheme.

        Args:
            op1 (Ciphertext): First polynomial argument (Multiplicand).
            op2 (Plaintext): Second polynomial argument (Multiplier).

        Returns:
            A Ciphertext object with a value equivalent to the result of the product of two
                operands.
        """
        ct, pt = ct.data, pt.data
        result = copy.deepcopy(ct)

        for i in range(len(result)):
            for j in range(len(self.coeff_modulus)):
                result[i][j] = poly_mul_mod(ct[i][0], pt, self.coeff_modulus[j], self.poly_modulus)

        return CipherText(result)

    def _mul_plain_plain(self, pt1, pt2):
        """Multiply two operands using FV scheme.

        Args:
            op1 (Plaintext): First polynomial argument (Multiplicand).
            op2 (Plaintext): Second polynomial argument (Multiplier).

        Returns:
            A Ciphertext object with a value equivalent to the result of the product of two
                operands.
        """
        pt1, pt2 = pt1.data, pt2.data
        return PlainText(poly_mul_mod(pt1, pt2, self.plain_modulus, self.poly_modulus))
