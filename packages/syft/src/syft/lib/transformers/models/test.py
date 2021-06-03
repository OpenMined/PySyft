from torch import nn
from syft import  SySequential, SyModule, serialize
import matplotlib.pyplot as plt

class Wrapper(SyModule):
    def __init__(self, module, input_size=(1, 10)):
        super().__init__(input_size=input_size)
        self.module = module
        
    def forward(self, x): return self.module(x=x)[0]
    
def make_nested_model(n):
    model = SySequential(nn.Linear(10, 10), input_size=(1, 10))
    for j in range(n):
        model = Wrapper(model)
    return model

bytesizes = []
for i in range(10):
    model = make_nested_model(i)
    bytesizes.append(serialize(model).ByteSize())
    
plt.plot(bytesizes)
plt.show()