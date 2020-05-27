from syft.frameworks.torch.he.fv.util.rns_tool import RNSTool


class Context:
    """A class used as for holding and easily supplying of all the general
    parameters required throughout the implementation.

    Attributes:
        param: An EncryptionParams object.
        coeff_div_plain_modulus: A list of float values equal to (q[i]/t),
            In research papers denoted by delta.
        rns_tool: A RNSTool class instance.
    """

    def __init__(self, encryption_param):

        self.param = encryption_param

        self.coeff_div_plain_modulus = [
            x / encryption_param.plain_modulus for x in encryption_param.coeff_modulus
        ]

        self.rns_tool = RNSTool(encryption_param)
