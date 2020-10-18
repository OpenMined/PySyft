from syft.frameworks.torch.he.fv.secret_key import SecretKey
from syft.frameworks.torch.he.fv.public_key import PublicKey
from syft.frameworks.torch.he.fv.ciphertext import CipherText
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
        self.context = context
        self.key = key

    def encrypt(self, message):
        """Encrypts an Plaintext data using the FV HE Scheme.

        Args:
            message (Plaintext): An plaintext which has to be encrypted.

        Retruns:
            A Ciphertext object containing the encrypted result.

        Raises:
            ValueError: Key provided for encryption is not a valid key object.
        """

        if isinstance(self.key, PublicKey):
            return self._encrypt(message, self.context.first_param_id, True)

        elif isinstance(self.key, SecretKey):
            return self._encrypt(message, self.context.first_param_id, False)

        else:
            raise ValueError("key for encryption is not valid")

    def _encrypt(self, message, param_id, is_asymmetric):
        """Encrypts the message according to the key provided.

        Args:
            message (Plaintext): An Plaintext object which has to be encrypted.
            param_id: Parameter id for accessing the correct parameters from the context chain.
            is_asymmetric (bool): Based on the key provided for encryption select
                the mode of encryption.

        Returns:
            A Ciphertext object contating the encrypted result.
        """

        result = None
        if is_asymmetric:
            prev_context_id = self.context.context_data_map[param_id].prev_context_id

            if prev_context_id is not None:
                # Requires modulus switching
                prev_context_data = self.context.context_data_map[prev_context_id]
                rns_tool = prev_context_data.rns_tool
                temp = encrypt_asymmetric(self.context, prev_context_id, self.key.data).data

                for j in range(2):
                    temp[j] = rns_tool.divide_and_round_q_last_inplace(temp[j])
                result = CipherText(temp, param_id)
            else:
                result = encrypt_asymmetric(
                    self.context, param_id, self.key.data
                )  # Public key used for encryption

        else:
            result = encrypt_symmetric(
                self.context, param_id, self.key.data
            )  # Secret key used for encryption

        return multiply_add_plain_with_delta(
            result, message, self.context.context_data_map[self.context.first_param_id]
        )
