import numpy as np
from syft.frameworks.torch.he.fv.util.rns_tool import RNSTool


class Context:
    """A class used as for holding and easily supplying of all the general parameters required throughout the implementation.

    Attributes:
        param: An EncryptionParams object.
        coeff_div_plain_modulus: A list of float values equal to (q[i]/t), In research papers denoted by delta.
        rns_tool: A RNSTool class instance.
    """

    def __init__(self, encryption_param):
        self._param = encryption_param
        self._coeff_div_plain_modulus = [
            x / encryption_param.plain_modulus for x in encryption_param.coeff_modulus
        ]
        self._rns_tool = RNSTool(
            self._param.poly_modulus_degree,
            encryption_param.coeff_modulus,
            encryption_param.plain_modulus,
        )

    @property
    def param(self):
        return self._param

    @property
    def coeff_div_plain_modulus(self):
        return self._coeff_div_plain_modulus

    @property
    def rns_tool(self):
        return self._rns_tool
