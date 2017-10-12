import numpy as np


class PolyApproximator:

    def __init__(self, f_real, degree=10, precision=10, min_range=-10, max_range=10):

        # interval over which we wish to optimize
        interval = np.linspace(min_range, max_range, 100)

        # interpolate polynomial of given max degree
        coefs = np.polyfit(interval, f_real(interval), degree)

        # reduce precision of interpolated coefficients
        self.coefs = [int(x * 10**precision) / 10**precision for x in coefs]

        # approximation function
        self.output = np.poly1d(self.coefs)
