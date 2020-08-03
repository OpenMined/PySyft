import copy
from enum import Enum

from syft.frameworks.torch.he.fv.util.operations import poly_add_mod
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

        if len(ct1) > 2 or len(ct2) > 2:
            # TODO: perform relinearization operation.
            raise Warning("Multiplying ciphertext of size >2 should be avoided.")

        rns_tool = self.context.rns_tool
        bsk_base_mod = rns_tool.base_Bsk.base
        bsk_base_mod_count = len(bsk_base_mod)
        dest_count = len(ct2) + len(ct1) - 1

        # Step 0: fast base convert from q to Bsk U {m_tilde}
        # Step 1: reduce q-overflows in Bsk
        # Iterate over all the ciphertexts inside ct1
        tmp_encrypted1_bsk = []
        for i in range(len(ct1)):
            tmp_encrypted1_bsk.append(rns_tool.sm_mrq(rns_tool.fastbconv_m_tilde(ct1[i])))

        # Similarly, itterate over all the ciphertexts inside ct2
        tmp_encrypted2_bsk = []
        for i in range(len(ct2)):
            tmp_encrypted2_bsk.append(rns_tool.sm_mrq(rns_tool.fastbconv_m_tilde(ct2[i])))

        tmp_des_coeff_base = [
            [[0] for y in range(len(self.coeff_modulus))] for x in range(dest_count)
        ]

        tmp_des_bsk_base = [[[0] for y in range(bsk_base_mod_count)] for x in range(dest_count)]

        for m in range(dest_count):
            # Loop over encrypted1 components [i], seeing if a match exists with an encrypted2
            # component [j] such that [i+j]=[m]
            # Only need to check encrypted1 components up to and including [m],
            # and strictly less than [encrypted_array.size()]

            current_encrypted1_limit = min(len(ct1), m + 1)

            for encrypted1_index in range(current_encrypted1_limit):
                # check if a corresponding component in encrypted2 exists
                if len(ct2) > m - encrypted1_index:
                    encrypted2_index = m - encrypted1_index

                    for i in range(len(self.coeff_modulus)):
                        tmp_des_coeff_base[m][i] = poly_add_mod(
                            poly_mul_mod(
                                ct1[encrypted1_index][i],
                                ct2[encrypted2_index][i],
                                self.coeff_modulus[i],
                                self.poly_modulus,
                            ),
                            tmp_des_coeff_base[m][i],
                            self.coeff_modulus[i],
                            self.poly_modulus,
                        )

                    for i in range(bsk_base_mod_count):
                        tmp_des_bsk_base[m][i] = poly_add_mod(
                            poly_mul_mod(
                                tmp_encrypted1_bsk[encrypted1_index][i],
                                tmp_encrypted2_bsk[encrypted2_index][i],
                                bsk_base_mod[i],
                                self.poly_modulus,
                            ),
                            tmp_des_bsk_base[m][i],
                            bsk_base_mod[i],
                            self.poly_modulus,
                        )

        # Now we multiply plain modulus to both results in base q and Bsk and
        # allocate them together in one container as
        # (te0)q(te'0)Bsk | ... |te count)q (te' count)Bsk to make it ready for
        # fast_floor
        result = []
        for i in range(dest_count):
            temp = []
            for j in range(len(self.coeff_modulus)):
                temp.append(
                    [
                        (x * self.plain_modulus) % self.coeff_modulus[j]
                        for x in tmp_des_coeff_base[i][j]
                    ]
                )
            for j in range(bsk_base_mod_count):
                temp.append(
                    [(x * self.plain_modulus) % bsk_base_mod[j] for x in tmp_des_bsk_base[i][j]]
                )
            result.append(rns_tool.fastbconv_sk(rns_tool.fast_floor(temp)))

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
