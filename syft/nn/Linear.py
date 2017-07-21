from syft import he
import numpy as np

class LinearClassifier():

    def __init__(self,n_inputs=4,n_labels=2,desc=""):
        super(type(LinearClassifier))

        self.desc = desc

        self.n_inputs = n_inputs
        self.n_labels = n_labels

        self.weights = list()
        for i in range(n_inputs):
            self.weights.append(np.zeros(n_labels).astype('float64'))

        self.pubkey = None

    def encrypt(self,pubkey):
        self.pubkey = pubkey
        for i,w in enumerate(self.weights):
            self.weights[i] = self.pubkey.encrypt(w)
        return self

    def decrypt(self,seckey):
        for i,w in enumerate(self.weights):
            self.weights[i] = seckey.decrypt(w)
        return self

    def forward(self,input):

        pred = self.weights[0] * input[0]
        for j,each_inp in enumerate(input[1:]):
            if(each_inp == 1):
                pred = pred + self.weights[j+1]
            elif(each_inp != 0):
                pred = pred + (self.weights[j+1] * input[j+1])

        return pred

    def learn(self,input,target,alpha=0.5):

        target = np.array(target).astype('float64')
        pred = self.forward(input)

        target_v = target

        if(self.pubkey is not None):
            target_v = self.pubkey.encrypt(target_v)

        grad = (pred - target_v) * alpha


        for i in range(len(input)):
            if(input[i] != 1 and input[i] != 0):
                self.weights[i] = self.weights[i] - (grad * input[i])
            elif(input[i] == 1):
                self.weights[i] = self.weights[i] - grad
            else:
                "doesn't matter... input == 0"

        return grad
