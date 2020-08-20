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
from syft.frameworks.torch.he.fv.relin_keys import RelinKey


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
        param = context.context_data_map[context.key_param_id].param
        self.plain_modulus = param.plain_modulus
        self.poly_modulus = param.poly_modulus

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
        context_data = self.context.context_data_map[ct.param_id]
        coeff_modulus = context_data.param.coeff_modulus
        result = copy.deepcopy(ct.data)

        for i in range(len(result)):
            for j in range(len(coeff_modulus)):
                for k in range(len(result[i][j])):
                    result[i][j][k] = negate_mod(result[i][j][k], coeff_modulus[j])

        return CipherText(result, ct.param_id)

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
        param_id = ct1.param_id
        context_data = self.context.context_data_map[param_id]
        coeff_modulus = context_data.param.coeff_modulus

        ct1, ct2 = copy.deepcopy(ct1.data), copy.deepcopy(ct2.data)
        result = ct2 if len(ct2) > len(ct1) else ct1

        for i in range(min(len(ct1), len(ct2))):
            for j in range(len(coeff_modulus)):
                result[i][j] = poly_add_mod(
                    ct1[i][j], ct2[i][j], coeff_modulus[j], self.poly_modulus
                )

        return CipherText(result, param_id)

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
        return multiply_add_plain_with_delta(ct, pt, self.context.context_data_map[ct.param_id])

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
        return multiply_sub_plain_with_delta(ct, pt, self.context.context_data_map[ct.param_id])

    def _sub_cipher_cipher(self, ct1, ct2):
        """Subtract two ciphertexts.

        Args:
            ct1 (Ciphertext): First polynomial argument (Minuend).
            ct2 (Ciphertext): Second polynomial argument (Subtrahend).

        Returns:
            A Ciphertext object with value equivalent to result of subtraction of two provided
                arguments.
        """
        param_id = ct1.param_id
        context_data = self.context.context_data_map[ct1.param_id]
        coeff_modulus = context_data.param.coeff_modulus
        ct1, ct2 = copy.deepcopy(ct1.data), copy.deepcopy(ct2.data)
        result = ct2 if len(ct2) > len(ct1) else ct1
        min_size, max_size = min(len(ct1), len(ct2)), max(len(ct1), len(ct2))

        for i in range(min_size):
            for j in range(len(coeff_modulus)):
                result[i][j] = poly_sub_mod(
                    ct1[i][j], ct2[i][j], coeff_modulus[j], self.poly_modulus
                )

        for i in range(min_size + 1, max_size):
            for j in range(len(coeff_modulus)):
                result[i][j] = poly_negate_mod(result[i][j], coeff_modulus[j])

        return CipherText(result, param_id)

    def _mul_cipher_cipher(self, ct1, ct2):
        """Multiply two operands using FV scheme.

        Args:
            op1 (Ciphertext): First polynomial argument (Multiplicand).
            op2 (Ciphertext): Second polynomial argument (Multiplier).

        Returns:
            A Ciphertext object with a value equivalent to the result of the product of two
                operands.
        """
        param_id = ct1.param_id
        context_data = self.context.context_data_map[param_id]
        coeff_modulus = context_data.param.coeff_modulus
        ct1, ct2 = ct1.data, ct2.data

        if len(ct1) > 2 or len(ct2) > 2:
            # TODO: perform relinearization operation.
            raise Warning("Multiplying ciphertext of size >2 should be avoided.")

        rns_tool = context_data.rns_tool
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

        tmp_des_coeff_base = [[[0] for y in range(len(coeff_modulus))] for x in range(dest_count)]

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

                    for i in range(len(coeff_modulus)):
                        tmp_des_coeff_base[m][i] = poly_add_mod(
                            poly_mul_mod(
                                ct1[encrypted1_index][i],
                                ct2[encrypted2_index][i],
                                coeff_modulus[i],
                                self.poly_modulus,
                            ),
                            tmp_des_coeff_base[m][i],
                            coeff_modulus[i],
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
            for j in range(len(coeff_modulus)):
                temp.append(
                    [(x * self.plain_modulus) % coeff_modulus[j] for x in tmp_des_coeff_base[i][j]]
                )
            for j in range(bsk_base_mod_count):
                temp.append(
                    [(x * self.plain_modulus) % bsk_base_mod[j] for x in tmp_des_bsk_base[i][j]]
                )
            result.append(rns_tool.fastbconv_sk(rns_tool.fast_floor(temp)))

        return CipherText(result, param_id)

    def _mul_cipher_plain(self, ct, pt):
        """Multiply two operands using FV scheme.
        Args:
            op1 (Ciphertext): First polynomial argument (Multiplicand).
            op2 (Plaintext): Second polynomial argument (Multiplier).
        Returns:
            A Ciphertext object with a value equivalent to the result of the product of two
                operands.
        """
        param_id = ct.param_id
        context_data = self.context.context_data_map[param_id]
        coeff_modulus = context_data.param.coeff_modulus
        ct, pt = ct.data, pt.data
        result = copy.deepcopy(ct)

        for i in range(len(result)):
            for j in range(len(coeff_modulus)):
                result[i][j] = poly_mul_mod(ct[i][0], pt, coeff_modulus[j], self.poly_modulus)

        return CipherText(result, param_id)

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

    def relin(self, ct, key):
        """Relinearize the provided ciphertext and decrease its size by one.
        (cannot apply on size 2 ciphetext)

        Args:
            ct (CipherText): ciphertext of size 3 to relinearize.
            key (Relin Key): relinearization key generated with keygenerator.

        Returns:
            Ciphertext of Size 2 with same encrypted value.
        """
        if len(ct.data) == 2:
            raise Warning("Ciphertext of size 2 does not need to relinearize.")
        if len(ct.data) > 3:
            raise Warning(f"Ciphertext of size {len(ct.data)} cannot be relinearized.")

        return self._switch_key_inplace(copy.deepcopy(ct), key)

    def _switch_key_inplace(self, ct, key):

        if not isinstance(key, RelinKey):
            raise RuntimeError("Relinearization key is invalid")

        param_id = ct.param_id
        ct = ct.data
        key_vector = key.data
        context_data = self.context.context_data_map[param_id]
        key_context = self.context.context_data_map[self.context.key_param_id]

        coeff_modulus = context_data.param.coeff_modulus
        decomp_mod_count = len(coeff_modulus)
        key_mod = key_context.param.coeff_modulus
        key_mod_count = len(key_mod)
        rns_mod_count = decomp_mod_count + 1

        target = ct[-1]  # Last component of ciphertext

        modswitch_factors = key_context.rns_tool.inv_q_last_mod_q

        for i in range(decomp_mod_count):

            local_small_poly_0 = copy.deepcopy(target[i])

            temp_poly = [[[0] for x in range(rns_mod_count)], [[0] for x in range(rns_mod_count)]]

            for j in range(rns_mod_count):
                index = key_mod_count - 1 if j == decomp_mod_count else j

                if key_mod[i] <= key_mod[index]:
                    local_small_poly_1 = copy.deepcopy(local_small_poly_0)
                else:
                    local_small_poly_1 = [x % key_mod[index] for x in local_small_poly_0]

                for k in range(2):
                    local_small_poly_2 = poly_mul_mod(
                        local_small_poly_1,
                        key_vector[i][k][index],
                        key_mod[index],
                        self.poly_modulus,
                    )
                    temp_poly[k][j] = poly_add_mod(
                        local_small_poly_2, temp_poly[k][j], key_mod[index], self.poly_modulus
                    )

        # Results are now stored in temp_poly[k]
        # Modulus switching should be performed
        for k in range(2):
            temp_poly_ptr = temp_poly[k][decomp_mod_count]
            temp_last_poly_ptr = temp_poly[k][decomp_mod_count]

            temp_poly_ptr = [x % key_mod[-1] for x in temp_poly_ptr]

            # Add (p-1)/2 to change from flooring to rounding.
            half = key_mod[-1] >> 1
            temp_last_poly_ptr = [(x + half) % key_mod[-1] for x in temp_last_poly_ptr]

            encrypted_ptr = ct[k]
            for j in range(decomp_mod_count):
                temp_poly_ptr = temp_poly[k][j]

                temp_poly_ptr = [x % key_mod[j] for x in temp_poly_ptr]
                local_small_poly = [x % key_mod[j] for x in temp_last_poly_ptr]
                half_mod = half % key_mod[j]

                local_small_poly = [(x - half_mod) % key_mod[j] for x in local_small_poly]

                # ((ct mod qi) - (ct mod qk)) mod qi
                temp_poly_ptr = poly_sub_mod(
                    temp_poly_ptr, local_small_poly, key_mod[j], self.poly_modulus
                )

                # qk^(-1) * ((ct mod qi) - (ct mod qk)) mod qi
                temp_poly_ptr = [(x * modswitch_factors[j]) % key_mod[j] for x in temp_poly_ptr]

                encrypted_ptr[j] = poly_add_mod(
                    temp_poly_ptr, encrypted_ptr[j], key_mod[j], self.poly_modulus
                )

        return CipherText(ct[0:2], param_id)
