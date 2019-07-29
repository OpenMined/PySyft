import torch
from .gradients_core import GradFunc
from .gradients_core import apply_dim_transformations


class AddBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self = grad.copy()
        grad_other = grad.copy() if type(self.self_) == type(self.other) else None

        if self.self_.shape != self.other.shape:
            grad_self, grad_other = apply_dim_transformations(
                grad_self, grad_other, self.self_.shape, self.other.shape
            )

        return (grad_self, grad_other)


class SubBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self = grad.copy()
        grad_other = grad * -1 if type(self.self_) == type(self.other) else None

        if self.self_.shape != self.other.shape:
            grad_self, grad_other = apply_dim_transformations(
                grad_self, grad_other, self.self_.shape, self.other.shape
            )
        return (grad_self, grad_other)


class SumBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        return ((self.self_ * 0 + 1) * grad,)
        # return (torch.ones(self.self_.shape) * grad, )


class AsinBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (-self.self_ * self.self_ + 1).rsqrt()
        return (grad_self_,)


class MulBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad * self.other
        grad_other = grad * self.self_ if type(self.self_) == type(self.other) else None
        return (grad_self_, grad_other)


class DivBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        assert isinstance(self.other, int)
        grad_self_ = grad / self.other
        return (grad_self_,)


class PowBackward(GradFunc):
    def __init__(self, self_, power):
        super().__init__(self, self_, power)
        self.self_ = self_
        self.power = power

    def gradient(self, grad):
        power = self.power
        return (power * self.self_ ** (power - 1) * grad,)


class MatmulBackward(GradFunc):
    def __init__(self, self_, other):
        super().__init__(self, self_, other)
        self.self_ = self_
        self.other = other

    def gradient(self, grad):
        grad_self_ = grad @ self.other.t()
        grad_other = self.self_.t() @ grad if type(self.self_) == type(self.other) else None
        return (grad_self_, grad_other)


class TBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        return (grad.t(),)


class SigmoidBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.sigmoid() * (1 - self.self_.sigmoid())
        return (grad_self_,)


class SinBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.cos()
        return (grad_self_,)


class SinhBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * self.self_.cosh()
        return (grad_self_,)


class SqrtBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad / (2 * self.result)
        return (grad_self_,)


class TanhBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        grad_self_ = grad * (1 - self.self_.tanh() ** 2)
        return (grad_self_,)


class ReluBackward(GradFunc):
    def __init__(self, self_):
        super().__init__(self, self_)
        self.self_ = self_

    def gradient(self, grad):
        zero = self.self_ * 0
        gt_zero = self.self_ > zero
        return (gt_zero * grad,)
