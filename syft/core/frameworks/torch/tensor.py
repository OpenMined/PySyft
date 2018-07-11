import json
import torch
import syft as sy

class _SyftTensor(object):
    ""
    
    def __init__(self, children):
        if(isinstance(children, list)):
            self.children = children
        else:
            self.children = [children]
    
    # def __str__(self):
        # return "blah"

    def ser(self, include_data=True, *args, **kwargs):

        tensor_msg = {}
        tensor_msg['type'] = str(self.__class__).split("'")[1]
        if hasattr(self, 'children'):
            tensor_msg['children'] = list(map(lambda child: child.ser(include_data=include_data,
                                                                      stop_recurse_at_torch_type=True),
                                              self.children))
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

        if('children' in msg_obj):
            children, leaf = _SyftTensor.deser(msg_obj['children'], highest_level=False)
            obj = obj_type(children)
            obj.id = msg_obj['id']
        elif('data' in msg_obj):
            obj = obj_type(msg_obj['data'])
            return obj, obj
        else:
            print("something went wrong...")
        if(highest_level):
            leaf.child = obj
            return leaf
        
        return obj, leaf

        


class _LocalTensor(_SyftTensor):

    def __init__(self, children):
        super().__init__(children=children)
        
    def __add__(self, other):
        """
        An example of how to overload a specific function given that 
        the default behavior in LocalTensor (for all other operations)
        is to simply call the native PyTorch functionality.
        """
        
        # custom stuff we can add
        print("adding2")
        
        # calling the native PyTorch functionality at the end
        
        # Note: for some reason you can't just call child.native___add__
        # because it will give an error that the function doesn't
        # exist. This seems to work fine though.
        results = list()
        for child in self.children:
            results.append(sy.Tensor.native___add__(child,other))

        if(len(results) == 1):
            return results[0]
        else:
            return results



class _PointerTensor(_SyftTensor):
    
    def __init__(self, children):
        super().__init__(children=children)
        
class _FixedPrecisionTensor(_SyftTensor):
    
    def __init__(self):
        super().__init__(children=children)

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

    def send(self, workers):

        workers = self.owners[0]._check_workers(self, workers)

        for worker in workers:
            self.owners[0].send_obj(self,
                                    worker,
                                    delete_local=True)

        self.set_(sy.zeros(0))

        self.children = [sy._PointerTensor(children=workers)]

        return self

    def ser(self, include_data=True, stop_recurse_at_torch_type=False):
        """Serializes a {} object to JSON.""".format(type(self))
        if(not stop_recurse_at_torch_type):
            serializations = list(map(lambda child: child.ser(include_data=include_data), self.children))
            return json.dumps(serializations) + "\n"
        else:
            tensor_msg = {}
            tensor_msg['type'] = str(self.__class__).split("'")[1]
            if include_data:
                tensor_msg['data'] = self.tolist()

            return tensor_msg

guard = {
    'syft.core.frameworks.torch.tensor._PointerTensor': _PointerTensor,
    'syft.core.frameworks.torch.tensor._SyftTensor': _SyftTensor,
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
