class SigmoidInterpolated10:

    def __init__(self):
        ONE = 1
        W0  = 0.5
        W1  = 0.2159198015
        W3  = -0.0082176259
        W5  = 0.0001825597
        W7  = -0.0000018848
        W9  = 0.0000000072
        self.sigmoid = np.vectorize(lambda x: \
            W0 + (x * W1) + (x**3 * W3) + (x**5 * W5) + (x**7 * W7) + (x**9 * W9))
        self.sigmoid_deriv = np.vectorize(lambda x:(ONE - x) * x)

    def forward(self, x):
        return self.sigmoid(x)

    def backward(self, x):
        return self.sigmoid_deriv(x)