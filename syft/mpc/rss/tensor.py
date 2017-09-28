from syft.tensor import TensorBase

import numpy as np


class RSSMPCTensor(TensorBase):

    def __init__(self, repo, data=None, input_is_shared=False):

        self.repo = repo
        self.encrypted = False

        if(input_is_shared):
            self.data = data
        else:

            if(type(data) == int or type(data) == float or type(data) == np.float64):
                self.data = self.repo.create_float(data)

            elif(type(data) == TensorBase):
                return RSSMPCTensor(self.repo, data.data)

            elif(type(data) == np.ndarray):
                x = data
                sh = x.shape
                x_ = x.reshape(-1)
                out = list()
                for v in x_:
                    out.append(self.repo.create_float(v))

                self.data = np.array(out).reshape(sh)
            else:
                print("format not recognized:" + str(type(x)))
                return NotImplemented

    def __add__(self, x):
        return RSSMPCTensor(self.repo, (self.data + x.data), True)

    def __mul__(self, x):
        return RSSMPCTensor(self.repo, (self.data * x.data), True)

    def __sub__(self, x):
        return RSSMPCTensor(self.repo, (self.data - x.data), True)

    def __str__(self):
        return "RSSMPCTensor: " + str(self.data)

    def __repr__(self):
        return "RSSMPCTensor: " + repr(self.data)

    def dot(self, x):
        return (self * x).sum(x.dim() - 1)
