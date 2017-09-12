import unittest

import numpy as np
from syft.he.paillier import KeyPair
from syft.nn.linear import LinearClassifier

import syft
from syft import TensorBase


class PySonarTests(unittest.TestCase):
    def hydrogeDemoNotebook(self):

        pubkey,prikey = KeyPair().generate(n_length=1024)
        diabetes_classifier = LinearClassifier(desc="DiabetesClassifier",n_inputs=10,n_labels=1)
        diabetes_classifier.encrypt(pubkey)

        self.assertTrue(True)
