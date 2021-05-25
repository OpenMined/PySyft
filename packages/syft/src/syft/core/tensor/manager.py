    
class ScalarChainManagerTensor():
    """Supports convenience methods for scalar chains of abstraction"""
    
    def push_abstraction_top(self, scalar_type, *args, **kwargs):
        ""
        
class TensorChainManager():
    
    def push_abstraction_top(self, tensor_type, *args, **kwargs):
        ""
        self.child = tensor_type(self.child, *args, **kwargs)