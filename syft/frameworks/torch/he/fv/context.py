class Context:
    """Stores a set of attributes (qualifiers) of a set of encryption parameters.
    These parameters are mainly used internally in various parts of the library,
    e.g., to determine which algorithmic optimizations the current support. The
    qualifiers are automatically created by the FVContext class, silently passed
    on to classes such as Encryptor, Evaluator, and Decryptor, and the only way to
    change them is by changing the encryption parameters themselves. In other
    words, a user will never have to create their own instance of this class, and
    in most cases never have to worry about it at all.
    """

    def __init__(self, encryption_param):
        self.param = encryption_param

    @property
    def param(self):
        return self.__param

    @param.setter
    def param(self, encryption_param):
        self.__param = encryption_param
