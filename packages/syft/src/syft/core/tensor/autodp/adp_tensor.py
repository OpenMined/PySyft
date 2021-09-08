class ADPTensor(object):
    """This tensor is just here to be an abstract class for now so that we can do
    isinstance() checks on DP tensors without having to worry about each one
    individually. We might add generic functionality to this in the future though."""
