class AbstractModel(object):
    """
        This is the interface used in all deep-learning models.
    """

    def encrypt(self, pubkey):
        return NotImplemented

    def decrypt(self, prikey):
        return NotImplemented

    def forward(self):
        return NotImplemented

    def learn(self, input, target, alpha=0.5, batchsize=32, encrypt_interval=16):
        return NotImplemented

    def batch_update(self, minibatch, alpha):
        return NotImplemented

    def evaluate(self, inputs, targets):
        return NotImplemented

    def generate_gradient(self, input, target):
        return NotImplemented

    def __repr__(self):
        return self.__str__()
