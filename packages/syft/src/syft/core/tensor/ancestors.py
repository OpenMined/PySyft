# stdlib
import uuid

# syft relative
from .manager import TensorChainManager
from .passthrough import is_acceptable_simple_type

_SingleEntityPhiTensorRef = None
def _SingleEntityPhiTensor():
    global _SingleEntityPhiTensorRef
    if _SingleEntityPhiTensorRef is None:
        # syft relative
        from .autodp.single_entity_phi import SingleEntityPhiTensor
        _SingleEntityPhiTensorRef = SingleEntityPhiTensor
    return _SingleEntityPhiTensorRef

_AutogradTensorRef = None
def _AutogradTensor():
    global _AutogradTensorRef
    if _AutogradTensorRef is None:
        # syft relative
        from .autograd.tensor import AutogradTensor
        _AutogradTensorRef = AutogradTensor
    return _AutogradTensorRef

class AutogradTensorAncestor(TensorChainManager):
    """Inherited by any class which might have or like to have AutogradTensor in its chain
    of .child objects"""
    
    @property
    def grad(self):
        return self.__class__(self.child.grad)
    
    @property
    def requires_grad(self):
        return self.child.requires_grad
    
    def backward(self, grad=None):

        AutogradTensor = _AutogradTensor()

        if isinstance(self.child, AutogradTensorAncestor) or isinstance(self.child, AutogradTensor):
            
            if grad is not None and not is_acceptable_simple_type(grad):
                grad = grad.child
            
            return self.child.backward(grad,backprop_id=uuid.uuid4())
        else:
            raise Exception("No AutogradTensor found in chain, but backward() method called.")
    
    def autograd(self, requires_grad=True):

        AutogradTensor = _AutogradTensor()

        self.push_abstraction_top(AutogradTensor, requires_grad=requires_grad)
        
        return self
    
    
class SingleEntityPhiTensorAncestor():
    """Inherited by any class which might have or like to have SingleEntityPhiTensor in its chain
    of .child objects"""

    @property
    def min_vals(self):
        return self.__class__(self.child.min_vals)

    @property
    def max_vals(self):
        return self.__class__(self.child.max_vals)

    def private(self, min_val, max_val, entities=None, entity=None):
        ""
        
        # if there's only one entity - push a SingleEntityPhiTensor
        child_type = _SingleEntityPhiTensor()
        entity=entity
        
        if isinstance(min_val, (float,int)):
            min_vals = (self.child * 0) + min_val
        else:
            raise Exception("min_val should be a float, got " + str(type(min_val)) + " instead.")
            
        if isinstance(max_val, (float,int)):
            max_vals = (self.child * 0) + max_val
        else:
            raise Exception("min_val should be a float, got " + str(type(min_val)) + " instead.")
            
        
        self.push_abstraction_top(child_type, entity=entity, min_vals=min_vals, max_vals=max_vals)
        
        # if there's row-level entities - push a MultiEntityPhiTensor
        
        
        # if there's element-level entities - push all elements with PhiScalars
        
        return self
