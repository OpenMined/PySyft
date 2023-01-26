# class NumericalTensor:
#     def __init__(self, child: Union[NumpyArray, ShareTensor, MockObject]):
#         self.child = child


# class AbstractDPMechanism:
#     def __init__(self, hyperparameters):
#         self.delta = hyperparameters['delta']

#     def private_transform(tensor, params):
#         pass

#     def publish(self, tensor: NumericalTensor, hyperparameters: dict[str, Any]):
#         private_output = self.private_transform(tensor, params)
#         self.side_effects(private_output)
#         return private_output.result
