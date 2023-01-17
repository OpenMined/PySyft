

class NumericalTensor:
    
    def __init__(child: Union[NumpyArray, ShareTensor, MockObject]):
        self.child = child


class AbstractDPMechanism:
    
    def __init__(hyperparameters):
        self.delta = hyperparameters['delta']
        
    def private_transform(tensor, params):
        
    
    def publish(tensor: NumericalTensor, hyperparameters: dict[str, Any]):
        private_output = self.private_transform(tensor, params)
        self.side_effects(private_output)
        return private_output.result