# third party
from ancestors import AutogradTensorAncestor
import numpy as np
from passthrough import PassthroughTensor
from passthrough import implements
from passthrough import inputs2child


class SingleEntityPhiTensor(PassthroughTensor, AutogradTensorAncestor):
    
    def __init__(self, child, entity, min_vals, max_vals):
        super().__init__(child)
        
        self.entity = entity
        self.min_vals = min_vals
        self.max_vals = max_vals
  
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
            
            print(min_min)
            
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
            
            print(min_min)
            print(max_max)
            
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
        
    def repeat(self, n):
        
        data = self.child.repeat(n)
        min_vals = self.min_vals.repeat(n)
        max_vals = self.max_vals.repeat(n)
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
    
    def sum(self, *args):
        
        data = self.child.sum(*args)
        min_vals = self.min_vals.sum(*args)
        max_vals = self.max_vals.sum(*args)
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
