import numpy as np
from passthrough import PassthroughTensor
from passthrough import implements
from passthrough import is_acceptable_simple_type
from passthrough import inputs2child
import uuid
from collections import Counter    
from collections import defaultdict
    
class AutogradTensor(PassthroughTensor):
        
    def __init__(self, child, requires_grad=False):
        super().__init__(child)
        
        # whether to run backpropagation or not        
        self.requires_grad = requires_grad
        
        # tensor gradient
        self._grad = defaultdict(lambda: None)
        
        # operation used to create this tensor (if any)
        self._grad_fn = None
        
        # list of ops which use this tensor
        self.ops = list()
        
        self.backprop_id = None
        
        self.n_backwards = Counter()
      
    @property
    def grad(self):
        if self.backprop_id not in self._grad:
            return None
        return self._grad[self.backprop_id]

    @property
    def grad_fn(self):
        if not self.requires_grad:
            raise Exception('This tensor is not backpropagated')
        return self._grad_fn    
    
#     def __ge__(self, other):
#         return AutogradTensor(self.child >= other.child, requires_grad=False)
    
#     def __le__(self, other):
#         return AutogradTensor(self.child <= other.child, requires_grad=False)    
    
    def __add__(self, other):
        
        op = AddOp()
        return op(self, other)
    
    def __sub__(self, other):
        op = SubOp()
        return op(self, other)    
    
    def __mul__(self, other):
        op = MulOp()
        return op(self, other)    
    
    def reshape(self, *shape):
        op = ReshapeOp()       
        return op(self, *shape)
    
    def copy(self):
        op = CopyOp()
        return op(self)
    
    def sum(self, *args, **kwargs):
        op = SumOp()
        return op(self, *args, **kwargs)
    
    def repeat(self, *args, **kwargs):
        op = RepeatOp()
        return op(self, *args, *kwargs)
    
    def transpose(self, *dims):
        op = TransposeOp()
        return op(self, *dims)
    
    def add_grad(self, grad):
        
        print("Adding grad:" + str(type(grad)) + " w/ backprop_id:" + str(self.backprop_id))
        
        if self._grad[self.backprop_id] is None:
            self._grad[self.backprop_id] = grad
        else:
            self._grad[self.backprop_id] = grad + self._grad[self.backprop_id]
            
    def backward(self, grad=None, backprop_id=None):        
        
        if backprop_id is None:
            backprop_id = uuid.uuid4()
            
        self.n_backwards[backprop_id] += 1

        self.backprop_id = backprop_id
    
        print("called backward on:" + str(self.grad_fn) + " " + str(self.n_backwards) + " " + str(self.ops))
    
        if not self.grad_fn:
            return False

        if grad is None and self._grad[self.backprop_id] is None:
            # in case if this is last loss tensor
            grad = np.ones(self.shape)
            grad = self.__class__(grad, requires_grad=False)

        elif self.grad is not None:
            grad = self._grad[self.backprop_id]

        if not self.requires_grad:
            raise Exception('This tensor is not backpropagated')
        
        print(self.n_backwards[backprop_id],len(self.ops))
        
        # if all gradients are accounted for - backprop
        if self.n_backwards[backprop_id] >= len(self.ops):
            
            self.grad_fn.backward(grad, backprop_id=backprop_id)
        
        # if some gradietns appear to be missing - parse forward in
        # the graph to double check
        else:
            
            # investigate whether any of the missing ops are actually
            # going to get used.
            found_id = False
            
            n_direct_ops = 0
            for op in self.ops:
                if op.backprop_id is not None and op.backprop_id == backprop_id:
                    n_direct_ops += 1
            
            # if the number of operations we know will be backpropagating gradietns to us
            # exceeds the number of times we've been backpropgated into - then we know
            # we need to wait.
            if n_direct_ops > self.n_backwards[backprop_id]:
                print(str(n_direct_ops) + " > " + str(self.n_backwards[backprop_id]))
                print(len(self.ops))
                found_id = True
            
            else:

                for op in self.ops:
                    if op.backprop_id is None:
                        if op.out.find_backprop_id(self.backprop_id):
                            found_id = True
                            break
                        
            print("Found id:"+ str(found_id))
            if found_id:
                "do nothing - we're going to get another gradient"
            else:
                # backprop anyway - we've got all the grads we're gonna get
                self.grad_fn.backward(grad, backprop_id=backprop_id)
            
        return True            
    
    def find_backprop_id(self, backprop_id):
        found_id = False
        
        for op in self.ops:
            if op.backprop_id is not None and op.backprop_id == backprop_id:
                return True

            if op.out.find_backprop_id(self.backprop_id):
                found_id = True
                break
        
        return found_id
        

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

@implements(AutogradTensor, np.expand_dims)
def expand_dims(*args, **kwargs):

    args, kwargs = inputs2child(*args, **kwargs)
    return np.expand_dims(*args, **kwargs)
    
class Op:
    
    def __init__(self):
        self.backprop_id = None

    def forward(self):
        raise NotImplemented

    def _backward(self, grad, backprop_id):
        raise NotImplemented

    def backward(self, grad, backprop_id):
        
        self.backprop_id = backprop_id
        
        for t in self.parent_tensors:
            t.backprop_id = backprop_id
        
        return self._backward(grad=grad, backprop_id=backprop_id)
        
    def __call__(self, *args, **kwargs):
        
        self.parent_tensors = list()
        
        for arg in args:
            if isinstance(arg, AutogradTensor):
                arg.ops.append(self)
                self.parent_tensors.append(arg)
                
        for key, arg in kwargs.items():
            if isinstance(arg, AutogradTensor):
                arg.ops.append(self)
                self.parent_tensor.append(arg)
        
        self.out = self.forward(*args, **kwargs)
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

    def _backward(self, grad, backprop_id):
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
                self.x.backward(backprop_id=backprop_id)
                
        if self.y.requires_grad:
            if self.y.shape != grad.shape:
                axis = np.argmax(np.abs(np.array(self.y.shape) - 
                                        np.array(grad.shape)))
                self.y.add_grad(AutogradTensor(grad.child.sum(axis=axis, 
                                                keepdims=True)))
            else:
                self.y.add_grad(grad)
            if self.y.grad_fn:
                self.y.backward(backprop_id=backprop_id)
                
class SubOp(Op):

    '''Substraction operation with 2 tensors'''

    def forward(self, x: AutogradTensor, y: AutogradTensor):
        self.x = x
        self.y = y
        
        requires_grad = x.requires_grad
        
        if is_acceptable_simple_type(y):
            return AutogradTensor(x.child - y, requires_grad=requires_grad)
        
        requires_grad = requires_grad or y.requires_grad
        return AutogradTensor(x.child - y.child, requires_grad=requires_grad)

    def _backward(self, grad, backprop_id):
        
        if self.x.requires_grad:
            # as we have matrix operation one of the parameters can 
            # have partial shape in such scenarion we need to sum
            # gradient values by missed axis
            if self.x.shape != grad.shape:
                print("shapes don't match")
                axis = np.argmax(np.abs(np.array(self.x.shape) - 
                                 np.array(grad.shape)))
                self.x.add_grad(AutogradTensor(grad.child.sum(axis=axis, 
                                                keepdims=True)))
            else:
                
                self.x.add_grad(grad)
                
            if self.x.grad_fn and self.y.grad_fn:
                self.x.backward(backprop_id=backprop_id)
                
        if self.y.requires_grad:
            if self.y.shape != grad.shape:
                print("shapes don't match")
                
                axis = np.argmax(np.abs(np.array(self.y.shape) - 
                                        np.array(grad.shape)))
                self.y.add_grad(AutogradTensor(-(grad.child.sum(axis=axis, 
                                                keepdims=True))))
            else:
                self.y.add_grad(-grad)
                
            if self.y.grad_fn and self.x.grad_fn:
                self.y.backward(backprop_id=backprop_id)
                                
                
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

    def _backward(self, grad, backprop_id):
        
        print("Sub backward grad:")
        print("Backprop id:" + str(len(self.x.ops)))
        print("Backprop id:" + str(len(self.y.ops)))
        print(grad)        
        
        y_is_simple = is_acceptable_simple_type(self.y)
        
        if self.x.requires_grad:
            
            if y_is_simple:
                self.x.add_grad(AutogradTensor(grad.child * self.y, False))
            else:            
                self.x.add_grad(AutogradTensor(grad.child * self.y.child, False))
                
            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
                
        if not y_is_simple and self.y.requires_grad:
            self.y.add_grad(AutogradTensor(grad.child * self.x.child, False))
            if self.y.grad_fn:
                self.y.backward(backprop_id=backprop_id)

class SumOp(Op):

    '''Sum operation across a dimension'''

    def forward(self, x: AutogradTensor, axis):
        self.x = x
        self.axis = axis
        self.dim_at_axis = self.x.shape[self.axis]
        self.backward_shape = self.x.shape

        return AutogradTensor(x.child.sum(axis), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):
        
        if self.x.requires_grad:
            
            requires_grad = grad.requires_grad

            grad = np.expand_dims(grad.child, self.axis)

            grad = grad.repeat(self.dim_at_axis, axis=self.axis)
                
            self.x.add_grad(AutogradTensor(grad, requires_grad=requires_grad))
                
            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)

                
class ReshapeOp(Op):

    '''Multiplication operation with 2 tensors'''

    def forward(self, x: AutogradTensor, *shape):
        self.x = x
        self.shape = shape
        self.backward_shape = self.x.shape

        return AutogradTensor(x.child.reshape(*shape), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):
        
        if self.x.requires_grad:
            
            self.x.add_grad(AutogradTensor(grad.child.reshape(self.backward_shape)))
                
            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id) 
                
                
class CopyOp(Op):

    '''Copy a tensor'''

    def forward(self, x: AutogradTensor):
        self.x = x

        return AutogradTensor(x.child.copy(), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):
        
        if self.x.requires_grad:
            
            self.x.add_grad(AutogradTensor(grad.child.copy()))
                
            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id) 
                                
                
class RepeatOp(Op):

    '''Repeat operation across a dimension'''

    def forward(self, x: AutogradTensor, repeats, axis=None):
        self.x = x
        self.repeats = repeats
        self.axis = axis
        
        self.input_shape = self.x.shape
        
        output = x.child.repeat(repeats=repeats, axis=axis)
        
        self.output_shape = output.shape
        
        return AutogradTensor(output, requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):
        
        if self.x.requires_grad:

            requires_grad = grad.requires_grad
                
            axis = self.axis
            if axis is None:
                axis = len(self.input_shape)-1
                
            intermediate_shape = list(self.input_shape)
            intermediate_shape.insert(axis+1, -1)

            grad = grad.child.reshape(intermediate_shape)
            
            grad = grad.sum(axis=axis+1)
                
            self.x.add_grad(AutogradTensor(grad, requires_grad=requires_grad))
                
            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
                
class TransposeOp(Op):

    '''Repeat operation across a dimension'''

    def forward(self, x:AutogradTensor, *dims):
        self.x = x
        self.dims = dims
        
        reverse_t_dims = {}
        for i, d in enumerate(self.dims):
            reverse_t_dims[d] = i

        l = sorted([(x[0],x[1]) for x in reverse_t_dims.items()], key=lambda x:x[0])
        self.reverse_dims = [x[1] for x in l]

        return AutogradTensor(x.child.transpose(*dims), requires_grad=x.requires_grad)

    def _backward(self, grad, backprop_id):
        
        if self.x.requires_grad:
            
            if y_is_simple:
                
                requires_grad = grad.requires_grad
                
                grad = grad.child.transpose(*self.reverse_dims)
                
                self.x.add_grad(AutogradTensor(grad, requires_grad=requires_grad))
                
            if self.x.grad_fn:
                self.x.backward(backprop_id=backprop_id)
                