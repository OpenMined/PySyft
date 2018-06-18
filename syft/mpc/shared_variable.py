import torch
from torch.autograd import Variable, Function
from . import spdz


class SharedAdd(Function):

    @staticmethod
    def forward(ctx, a, b):
        return spdz.spdz_add(a, b)

    @staticmethod
    def backward(ctx, grad_out):
        grad_out = grad_out
        return grad_out, grad_out


class SharedNeg(Function):
    
    @staticmethod
    def forward(ctx,a):
        return spdz.spdz_neg(a)
    
    @staticmethod
    def backward(ctx,grad_out):
        return spdz.spdz_neg(grad_out)


class SharedSub(Function):
    
    @staticmethod
    def forward(ctx,a,b):
        return spdz.spdz_add(a,spdz.spdz_neg(b))
    
    @staticmethod
    def backward(ctx,grad_out):
        return grad_out, spdz.spdz_neg(grad_out)


class SharedMult(Function):

    @staticmethod
    def forward(ctx, a, b, interface):
        ctx.save_for_backward(a, b)
        ctx.interface = interface
        return spdz.spdz_mul(a, b, interface)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        interface = ctx.interface
        grad_out = grad_out
        return Variable(spdz.spdz_mul(grad_out.data, b, interface)), Variable(spdz.spdz_mul(grad_out.data, a, interface)),None


class SharedMatmul(Function):

    @staticmethod
    def forward(ctx, a, b, interface):
        ctx.save_for_backward(a, b)
        ctx.interface = interface
        return spdz.spdz_matmul(a, b, interface)

    @staticmethod
    def backward(ctx, grad_out):
        a, b = ctx.saved_tensors
        interface = ctx.interface
        grad_out = grad_out.data
        a_grad = Variable(spdz.spdz_matmul(grad_out,  b.t_(), interface))
        b_grad = Variable(spdz.spdz_matmul( a.t_(),grad_out, interface)) 
        return a_grad,b_grad ,None


class SharedSigmoid(Function):

    @staticmethod
    def forward(ctx, a, interface):
        ctx.save_for_backwards(a)
        ctx.interface = interface
        return spdz.spdz_sigmoid(a, interface)

    @staticmethod
    def backward(ctx, grad_out):
        a = ctx.saved_tensors
        interface = ctx.interface
        ones = spdz.encode(torch.FloatTensor(a.shape).one_())
        return spdz.spdz_mul(a, spdz.public_add(ones, -a, interface), interface)


class SharedVariable(object):

    def __init__(self, var, interface):
        if not isinstance(var, Variable):
            raise ValueError('Var must be a variable')
        else:
            self.var = var
        self.interface = interface

    def __neg__(self):
        return self.neg()

    def __add__(self, other):
        return self.add(other)
    
    def __sub__(self, other):
        return self.sub(other)

    def __mul__(self, other):
        return self.mul(other)

    def __matmul__(self, other):
        return self.matmul(other)

    def sigmoid(self):
        return SharedVariable(SharedSigmoid.apply(self.var, self.interface), self.interface)
    
    def neg(self):
        return SharedVariable(SharedNeg.apply(self.var),self.interface)
    
    def add(self, other):
        return SharedVariable(SharedAdd.apply(self.var, other.var), self.interface, self.requires_grad)
    
    def sub(self, other):
        return SharedVariable(SharedSub.apply(self.var,other.var),self.interface)

    def mul(self, other):
        return SharedVariable(SharedMult.apply(self.var, other.var, self.interface), self.interface)

    def matmul(self, other):
        return SharedVariable(SharedMatmul.apply(self.var, other.var, self.interface), self.interface)

    @property
    def grad(self):
        return self.var.grad

    @property
    def data(self):
        return self.var.data
    
    def backward(self,grad):
        return self.var.backward(grad)

    def t_(self):
        self.var = self.var.t_()

    def __repr__(self):
        return self.var.__repr__()

    def type(self):
        return 'SharedVariable'
