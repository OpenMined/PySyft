# third party
import numpy as np

# syft relative
from ..ancestors import AutogradTensorAncestor
from ..passthrough import PassthroughTensor
from ..passthrough import implements
from ..passthrough import inputs2child
from ..passthrough import is_acceptable_simple_type


class SingleEntityPhiTensor(PassthroughTensor, AutogradTensorAncestor):
    
    def __init__(self, child, entity, min_vals, max_vals):
        super().__init__(child)
        
        self.entity = entity
        self._min_vals = min_vals
        self._max_vals = max_vals

    @property
    def min_vals(self):
        return self._min_vals

    @property
    def max_vals(self):
        return self._max_vals

    def __add__(self, other):

        if isinstance(other, SingleEntityPhiTensor):
                    
            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented
            
            data = self.child + other.child
            min_vals = self.min_vals + other.min_vals
            max_vals = self.max_vals + other.max_vals
            entity = self.entity
            
            return SingleEntityPhiTensor(child=data,
                                         entity=entity,
                                         min_vals=min_vals,
                                         max_vals=max_vals)

        elif is_acceptable_simple_type(other):

            data = self.child + other
            min_vals = self.min_vals + other
            max_vals = self.max_vals + other
            entity = self.entity

            return SingleEntityPhiTensor(child=data,
                                         entity=entity,
                                         min_vals=min_vals,
                                         max_vals=max_vals)

        else:
            return NotImplemented
        
    def __sub__(self, other):

        if isinstance(other, SingleEntityPhiTensor):
                    
            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented
            
            data = self.child - other.child
            min_vals = self.min_vals - other.min_vals
            max_vals = self.max_vals - other.max_vals
            entity = self.entity
            
            return SingleEntityPhiTensor(child=data,
                                         entity=entity,
                                         min_vals=min_vals,
                                         max_vals=max_vals)
        else:
            return NotImplemented  
        
    def __mul__(self, other):

        if other.__class__ == SingleEntityPhiTensor:
                    
            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented
            
            data = self.child * other.child
            
            min_min = self.min_vals * other.min_vals
            min_max = self.min_vals * other.max_vals
            max_min = self.max_vals * other.min_vals
            max_max = self.max_vals * other.max_vals            

            min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)
            max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)
            entity = self.entity
            
            return SingleEntityPhiTensor(child=data,
                                         entity=entity,
                                         min_vals=min_vals,
                                         max_vals=max_vals)
        else:
            
            
            data = self.child * other
            
            min_min = self.min_vals * other
            max_max = self.max_vals * other            

            min_vals = np.min([min_min, max_max], axis=0)
            max_vals = np.max([min_min, max_max], axis=0)
            entity = self.entity
            
            return SingleEntityPhiTensor(child=data,
                                         entity=entity,
                                         min_vals=min_vals,
                                         max_vals=max_vals)
            
        
    def __truediv__(self, other):

        if isinstance(other, SingleEntityPhiTensor):
                    
            if self.entity != other.entity:
                # this should return a GammaTensor
                return NotImplemented
            
            data = self.child / other.child
            
            if (other.min_vals == 0).any() or (other.max_vals == 0).any():
            
                raise Exception("Infinite sensitivity - we can support this in the future but not yet")

            else:    

                min_min = self.min_vals / other.min_vals
                min_max = self.min_vals / other.max_vals
                max_min = self.max_vals / other.min_vals
                max_max = self.max_vals / other.max_vals            

                min_vals = np.min([min_min, min_max, max_min, max_max], axis=0)
                max_vals = np.max([min_min, min_max, max_min, max_max], axis=0)
                
            entity = self.entity
            
            return SingleEntityPhiTensor(child=data,
                                         entity=entity,
                                         min_vals=min_vals,
                                         max_vals=max_vals)
        else:
            return self * (1 / other)         
        
    def repeat(self, repeats, axis=None):
        
        data = self.child.repeat(repeats, axis=axis)
        min_vals = self.min_vals.repeat(repeats, axis=axis)
        max_vals = self.max_vals.repeat(repeats, axis=axis)
        entity = self.entity
        
        return SingleEntityPhiTensor(child=data,
                                     entity=entity,
                                     min_vals=min_vals,
                                     max_vals=max_vals)  
    
    def reshape(self, *args):
        
        data = self.child.reshape(*args)
        min_vals = self.min_vals.reshape(*args)
        max_vals = self.max_vals.reshape(*args)
        entity = self.entity
        
        return SingleEntityPhiTensor(child=data,
                                     entity=entity,
                                     min_vals=min_vals,
                                     max_vals=max_vals)
    
    def sum(self, *args, **kwargs):
        
        data = self.child.sum(*args, **kwargs)
        min_vals = self.min_vals.sum(*args, **kwargs)
        max_vals = self.max_vals.sum(*args, **kwargs)
        entity = self.entity
        
        return SingleEntityPhiTensor(child=data,
                                     entity=entity,
                                     min_vals=min_vals,
                                     max_vals=max_vals)    
        

    def dot(self,  other):
        return self.manual_dot(other)
        
    def transpose(self, *args, **kwargs):
        
        data = self.child.transpose(*args)
        min_vals = self.min_vals.transpose(*args)
        max_vals = self.max_vals.transpose(*args)
        entity = self.entity
        
        return SingleEntityPhiTensor(child=data,
                                     entity=entity,
                                     min_vals=min_vals,
                                     max_vals=max_vals)    

# @implements(SingleEntityPhiTensor, np.min)
# def npmax(*args, **kwargs):
#     print("mining1")
#     args, kwargs = inputs2child(*args, **kwargs)
#     return np.min(*args, **kwargs)
    
@implements(SingleEntityPhiTensor, np.mean)
def mean(*args, **kwargs):

    entity = args[0].entity
    
    for arg in args[1:]:
        if not isinstance(arg, SingleEntityPhiTensor):
            raise Exception("Can only call np.mean on objects of the same type.")

        if arg.entity != entity:
            return NotImplemented
    
    min_vals = np.mean([x.min_vals for x in args], **kwargs)
    max_vals = np.mean([x.max_vals for x in args], **kwargs)
    
    args,kwargs = inputs2child(*args, **kwargs)
    
    data = np.mean(args, **kwargs)
    
    return SingleEntityPhiTensor(child=data,
                             entity=entity,
                             min_vals=min_vals,
                             max_vals=max_vals)


@implements(SingleEntityPhiTensor, np.expand_dims)
def expand_dims(a, axis):

    entity = a.entity

    min_vals = np.expand_dims(a=a.min_vals, axis=axis)
    max_vals = np.expand_dims(a=a.max_vals, axis=axis)

    data = np.expand_dims(a.child, axis=axis)

    return SingleEntityPhiTensor(child=data,
                                 entity=entity,
                                 min_vals=min_vals,
                                 max_vals=max_vals)