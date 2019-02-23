class GradAdd:
    def __init__(self, *args):
        self.next_functions = []
        
    def __call__(self, grad):
        return grad # * 1

class Accumulate:
    def __init__(self, tensor):
        self.tensor = tensor
        self.next_functions = []
        
    def __call__(self, grad):
        print(self.tensor)
        if self.tensor.grad is not None:
            self.tensor.grad += grad
        else:
            self.tensor.grad = grad + 0