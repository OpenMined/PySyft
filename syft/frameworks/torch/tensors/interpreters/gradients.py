# This file is generated from build_gradients.py

from . gradients_core import *

class AddBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad
        grad_other = grad if type(self.self_) == type(self.other) else None
        return (grad_self_, grad_other,)

class AsinBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (-self.self_ * self.self_ + 1).rsqrt()
        return (grad_self_,)

