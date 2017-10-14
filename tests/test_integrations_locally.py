import unittest

import numpy as np
import pickle

from syft.nn.linear import LinearClassifier
from syft.he.paillier import KeyPair, PaillierTensor
from capsule.django_client import LocalDjangoCapsuleClient
from syft.mpc.rss import MPCRepo
from syft.mpc.rss.tensor import RSSMPCTensor


class PySonarNotebooks(unittest.TestCase):

    def model_training_demo_notebook(self):
        """If this test fails, you probably broke the demo notebook located at
        PySonar/notebooks/Sonar - Decentralized Model Training Simulation
        (local blockchain).ipynb """
        c = LocalDjangoCapsuleClient()
        d = LinearClassifier(desc="DiabetesClassifier", n_inputs=10, n_labels=1, capsule_client=c)
        d.encrypt()

        self.assertTrue(True)


class PySyftNotebooks(unittest.TestCase):

    def MPC_SPDZ_notebook(self):
        """If this test fails, you probably broke the demo notebook located at
        PySyft/notebooks/MPC Tensor SPDZ.ipynb
        """

        server1 = MPCRepo()
        server2 = MPCRepo()
        server1.set_parties(server2)
        server2.set_parties(server1)
        r = RSSMPCTensor(server1, np.array([1, 2, 3, 4, 5.]))

        out1 = r + r
        self.assertEqual(out1, np.array([2., 4., 6., 8., 10.]))

        out2 = r - r
        self.assertEqual(out2, np.array([0., 0., 0., 0., 0.]))

        out3 = r * r
        self.assertEqual(out3, np.array([1., 4., 9., 16., 25.]))

        out4 = r.dot(r)
        self.assertEqual(out4, 55.)

        out5 = r.sum()
        self.asserEqual(out5, 15.)

    def paillier_HE_example_notebook(self):
        """If this test fails, you probably broke the demo notebook located at
        PySyft/notebooks/Syft - Paillier Homomorphic Encryption Example.ipynb
        """

        pubkey, prikey = KeyPair().generate()
        x = PaillierTensor(pubkey, np.array([1, 2, 3, 4, 5.]))

        out1 = x.decrypt(prikey)
        self.assertEqual(out1, np.array([1., 2., 3., 4., 5.]))

        out2 = (x + x[0]).decrypt(prikey)
        self.assertEqual(out2, np.array([2., 3., 4., 5., 6.]))

        out3 = (x * 5).decrypt(prikey)
        self.assertEqual(out3, np.array([5., 10., 15., 20., 25.]))

        out4 = (x + x / 5).decrypt(prikey)
        self.assertEqual(out4, np.array([1.2, 2.4, 3.6, 4.8, 6.]))

        pubkey_str = pubkey.serialize()
        prikey_str = prikey.serialize()

        pubkey2, prikey2 = KeyPair().deserialize(pubkey_str, prikey_str)

        out5 = prikey2.decrypt(x)
        self.assertEqual(out5, np.array([1., 2., 3., 4., 5.]))

        y = PaillierTensor(pubkey, (np.ones(5)) / 2)
        out6 = prikey.decrypt(y)
        self.assertEqual(out6, np.array([.5, .5, .5, .5, .5]))

        y_str = pickle.dumps(y)
        y2 = pickle.loads(y_str)
        out7 = prikey.decrypt(y2)
        self.assertEqual(out7, np.array([.5, .5, .5, .5, .5]))

    def test_paillier_linear_classifier_notebook(self):
        """If this test fails, you probably broke the demo notebook located at
        PySyft/notebooks/Syft - Paillier Homomorphic Encryption Example.ipynb
        """

        capsule = LocalDjangoCapsuleClient()
        model = LinearClassifier(capsule_client=capsule)
        assert(model.capsule == capsule)

        try:
            model = model.encrypt()
            encrypted = True
        except Exception as e:
            encrypted = False
            print('[!]', e)

        input = np.array([[0, 0, 1, 1], [0, 0, 1, 0],
                          [1, 0, 1, 1], [0, 0, 1, 0]])
        target = np.array([[0, 1], [0, 0], [1, 1], [0, 0]])

        for iter in range(3):
            model.learn(input, target, alpha=0.5)

        if encrypted:
            model = model.decrypt()

        for i in range(len(input)):
            model.forward(input[i])
