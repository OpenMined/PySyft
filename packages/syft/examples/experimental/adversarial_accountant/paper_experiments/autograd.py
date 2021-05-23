import numpy as np
from passthrough import PassthroughTensor
from passthrough import implements
from passthrough import is_acceptable_simple_type
from passthrough import inputs2child
        
class AutogradTensor(PassthroughTensor):
        
    def __init__(self, child, requires_grad=False):
        super().__init__(child)
        
        # whether to run backpropagation or not        
        self.requires_grad = requires_grad
        
        # tensor gradient
        self._grad = None
        
        # operation used to create this tensor (if any)
        self._grad_fn = None
      
    @property
    def grad(self):
        return self._grad

    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise Exception('This tensor is not backpropagated')
        return self._grad_fn    
    
    def __ge__(self, other):
        return AutogradTensor(self.child >= other.child, requires_grad=False)
    
    def __le__(self, other):
        return AutogradTensor(self.child <= other.child, requires_grad=False)    
    
    def __add__(self, other):
        return AddOp()(self, other)
    
    def __mul__(self, other):
        return MulOp()(self, other)    
    
    def reshape(self, *shape):
        return ReshapeOp()(self, *shape)
    
    def add_grad(self, grad):
        
        print("Adding grad:" + str(type(grad)))
        
        if self._grad is None:
            self._grad = grad
        else:
            self._grad = self._grad + grad    
    
    def backward(self, grad=None):
        if not self.grad_fn:
            return False

        if grad is None and self._grad is None:
            # in case if this is last loss tensor
            grad = self.__class__(1., requires_grad=False)

        elif self.grad is not None:
            grad = self._grad

        if not self.requires_grad:
            raise Exception('This tensor is not backpropagated')
            
        self.grad_fn.backward(grad)
        return True            

@implements(AutogradTensor, np.max)
def npmax(*args, **kwargs):
    print("maxing")
    args, kwargs = inputs2child(*args, **kwargs)
    return np.max(*args, **kwargs)
    
@implements(AutogradTensor, np.min)
def npmin(*args, **kwargs):
    print("mining")
    print(args)
    args, kwargs = inputs2child(*args, **kwargs)
    return np.min(*args, **kwargs)
    
class Op:

    def forward(self):
        raise NotImplemented

    def backward(self, grad):
        raise NotImplemented

    def __call__(self, *args):
        self.out = self.forward(*args)
        self.out._grad_fn = self
        return self.out

class AddOp(Op):

    '''Sumation operation with 2 tensors'''

    def forward(self, x: AutogradTensor, y: AutogradTensor):
        self.x = x
        self.y = y
        
        requires_grad = x.requires_grad
        
        if is_acceptable_simple_type(y):
            return AutogradTensor(x.child + y, requires_grad=requires_grad)
        
        requires_grad = requires_grad or y.requires_grad
        return AutogradTensor(x.child + y.child, requires_grad=requires_grad)

    def backward(self, grad):
        if self.x.requires_grad:
            # as we have matrix operation one of the parameters can 
            # have partial shape in such scenarion we need to sum
            # gradient values by missed axis
            if self.x.shape != grad.shape:
                axis = np.argmax(np.abs(np.array(self.x.shape) - 
                                 np.array(grad.shape)))
                self.x.add_grad(AutogradTensor(grad.child.sum(axis=axis, 
                                                keepdims=True)))
            else:
                self.x.add_grad(grad)
            if self.x.grad_fn:
                self.x.backward()
        if self.y.requires_grad:
            if self.y.shape != grad.shape:
                axis = np.argmax(np.abs(np.array(self.y.shape) - 
                                        np.array(grad.shape)))
                self.y.add_grad(AutogradTensor(grad.child.sum(axis=axis, 
                                                keepdims=True)))
            else:
                self.y.add_grad(grad)
            if self.y.grad_fn:
                self.y.backward()
                
class MulOp(Op):

    '''Multiplication operation with 2 tensors'''

    def forward(self, x: AutogradTensor, y: AutogradTensor):
        self.x = x
        self.y = y
        
        requires_grad = x.requires_grad
        
        if is_acceptable_simple_type(y):
            return AutogradTensor(x.child * y, requires_grad=requires_grad)
        
        requires_grad = requires_grad or y.requires_grad
        return AutogradTensor(x.child * y.child, requires_grad=requires_grad)

    def backward(self, grad):
        
        y_is_simple = is_acceptable_simple_type(self.y)
        
        if self.x.requires_grad:
            
            if y_is_simple:
                self.x.add_grad(AutogradTensor(grad.child * self.y, False))
            else:            
                self.x.add_grad(AutogradTensor(grad.child * self.y.child, False))
                
            if self.x.grad_fn:
                self.x.backward()
                
        if not y_is_simple and self.y.requires_grad:
            self.y.add_grad(AutogradTensor(grad.child * self.x.child, False))
            if self.y.grad_fn:
                self.y.backward()
                
class ReshapeOp(Op):

    '''Multiplication operation with 2 tensors'''

    def forward(self, x: AutogradTensor, *shape):
        self.x = x
        self.shape = shape
        self.backward_shape = self.x.shape

        return AutogradTensor(x.child.reshape(*shape), requires_grad=x.requires_grad)

    def backward(self, grad):
        
        if self.x.requires_grad:
            
            if y_is_simple:
                self.x.add_grad(AutogradTensor(*grad.child.reshape(self.backward_shape)))
                
            if self.x.grad_fn:
                self.x.backward()
                
# class SumOp(Op):

#     '''Multiplication operation with 2 tensors'''

#     def forward(self, x: AutogradTensor, *shape):
#         self.x = x
#         self.shape = shape
#         self.backward_shape = self.x.shape

#         return AutogradTensor(x.child.reshape(*shape), requires_grad=x.requires_grad)

#     def backward(self, grad):
        
#         if self.x.requires_grad:
            
#             if y_is_simple:
#                 self.x.add_grad(AutogradTensor(*grad.child.reshape(self.backward_shape)))
                
#             if self.x.grad_fn:
#                 self.x.backward()
                