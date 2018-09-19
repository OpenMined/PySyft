import syft as sy
from syft.core.frameworks.torch.tensor import _SyftTensor
from syft.core.frameworks.torch import torch_utils
class _GeneralizedPointerTensor(_SyftTensor):

    def __init__(self, pointer_tensor_dict, parent=None, torch_type=None, id=None, owner=None, skip_register=False):
         super().__init__(child=None, parent=parent, torch_type=torch_type, owner=owner, id=id,
                         skip_register=skip_register)
         pointer_dict = {}
         for worker, pointer in pointer_tensor_dict.items():
             if not isinstance(pointer, sy._PointerTensor):
                 raise TypeError('Should use sy._Pointer without Torch wrapper.')
             key = worker if isinstance(worker, (int, str)) else worker.id
             pointer_dict[key] = pointer
         self.pointer_tensor_dict = pointer_dict

    @classmethod
    def handle_call(cls, syft_command, owner):
        syft_commands = torch_utils.split_to_pointer_commands(syft_command)
        result_dict = {}
        for worker_id in syft_commands.keys():
            syft_command = syft_commands[worker_id]
            result_dict[worker_id] = sy._PointerTensor.handle_call(syft_command, owner)

        #TODO: @trask @theo could you take a look at this if you have better ideas on how to get these parameters
        gpt =  _GeneralizedPointerTensor(result_dict, None, None, id=None, owner=owner, skip_register=False)
        # Fixme: Add a generic child depending on a torch_type
        gpt.child = sy.FloatTensor([])
        return gpt

    def get(self):
        res = []
        for worker, pointer in self.pointer_tensor_dict.items():
            res.append(pointer.get())
        return res

    def sum_get(self):
        shares = self.get()
        res = None
        for share in shares:
            if res is None:
                res = share
            else:
                res += share
        return res
