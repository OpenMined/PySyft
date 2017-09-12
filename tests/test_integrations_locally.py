import unittest

from syft.he.paillier import KeyPair
from syft.nn.linear import LinearClassifier


class PySonarNotebooks(unittest.TestCase):
    def hydrogeDemoNotebook(self):
        """If this method fails, you probably broke the demo notebook located at
        PySonar/notebooks/Sonar - Decentralized Model Training Simulation
        (local blockchain).ipynb """

        pubkey, prikey = KeyPair().generate(n_length=1024)
        d = LinearClassifier(desc="DiabetesClassifier", n_inputs=10, n_labels=1)
        d.encrypt(pubkey)

        self.assertTrue(True)
