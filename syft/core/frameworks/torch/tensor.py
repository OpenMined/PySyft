import json
import torch

class _AbstractTensor(object):
    ""
    
    def __init__(self, child):
        self.child = child
    
    # def __str__(self):
        # return "blah"

    def ser(self, include_data=True, *args, **kwargs):

        tensor_msg = {}
        tensor_msg['type'] = str(self.__class__).split("'")[1]
        if hasattr(self, 'child'):
            tensor_msg['child'] = self.child.ser(include_data=include_data,
                                                 stop_recurse_at_torch_type=True)
        tensor_msg['id'] = self.id
        owner_type = type(self.owners[0])
        if (owner_type is int or owner_type is str):
            tensor_msg['owners'] = self.owners
        else:
            tensor_msg['owners'] = list(map(lambda x: x.id, self.owners))

        return tensor_msg

    @staticmethod
    def deser(msg, highest_level=True):
        if isinstance(msg, str):
            msg_obj = json.loads(msg)
        else:
            msg_obj = msg

        obj_type = guard[msg_obj['type']]

        if('child' in msg_obj):
            child, leaf = _AbstractTensor.deser(msg_obj['child'], highest_level=False)
            obj = obj_type(child)
        elif('data' in msg_obj):
            obj = obj_type(msg_obj['data'])
            return obj, obj
        else:
            print("something went wrong...")
        if(highest_level):
            leaf.child = obj
            return leaf
        
        return obj, leaf

        


class _LocalTensor(_AbstractTensor):

    def __init__(self, child):
        super().__init__(child=child)
        
    def __add__(self, other):
        """
        An example of how to overload a specific function given that 
        the default behavior in LocalTensor (for all other operations)
        is to simply call the native PyTorch functionality.
        """
        
        # custom stuff we can add
#         print("adding")
        
        # calling the native PyTorch functionality at the end
        return self.child.native___add__(other)

class _PointerTensor(_AbstractTensor):
    
    def __init__(self, child):
        super().__init__(child=child)
        
class _FixedPrecisionTensor(_AbstractTensor):
    
    def __init__(self):
        super().__init__(child=child)

class _TorchTensor(object):
    """
    This tensor is simply a more convenient way to add custom
    functions to all Torch tensor types.
    """

    __module__ = 'syft'

    def __str__(self):
        return self.native___str__()

    def __repr__(self):
        return self.native___repr__()

    def ser(self, include_data=True, stop_recurse_at_torch_type=False):
        """Serializes a {} object to JSON.""".format(type(self))
        if(not stop_recurse_at_torch_type):
            return json.dumps(self.child.ser(include_data=include_data)) + "\n"
        else:
            tensor_msg = {}
            tensor_msg['type'] = str(self.__class__).split("'")[1]
            if include_data:
                tensor_msg['data'] = self.tolist()

            return tensor_msg

guard = {
    'syft.core.frameworks.torch.tensor._PointerTensor': _PointerTensor,
    'syft.core.frameworks.torch.tensor._AbstractTensor': _AbstractTensor,
    'syft.core.frameworks.torch.tensor._LocalTensor': _LocalTensor,
    'syft.core.frameworks.torch.tensor._FixedPrecisionTensor': _FixedPrecisionTensor,
    'syft.FloatTensor': torch.FloatTensor,
    'syft.DoubleTensor': torch.DoubleTensor,
    'syft.HalfTensor': torch.HalfTensor,
    'syft.ByteTensor': torch.ByteTensor,
    'syft.CharTensor': torch.CharTensor,
    'syft.ShortTensor': torch.ShortTensor,
    'syft.IntTensor': torch.IntTensor,
    'syft.LongTensor': torch.LongTensor
}
