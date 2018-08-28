import torch
from unittest import TestCase

from syft.core.hooks import TorchHook
from syft.core.utils import PythonJSONDecoder


class TestPythonJSONDecoder(TestCase):
    def test_custom_obj_hook(self):
        hook = TorchHook()
        local = hook.local_worker
        objects = [ # (object, id)
            (torch.FloatTensor([0]), 1111),
            (torch.FloatTensor([1]), 'this_is_a_string_id'),
            (torch.autograd.Variable(torch.FloatTensor([2])), 2222),
            (torch.autograd.Variable(torch.FloatTensor([3])), 'three'),
            (torch.nn.Parameter(torch.FloatTensor([4])), 4444),
            (torch.nn.Parameter(torch.FloatTensor([5])), 'five'),
            (torch.DoubleTensor([6]), 6666),
            (torch.DoubleTensor([7]), 'seven'),
            (torch.HalfTensor([8]), 8888),
            (torch.HalfTensor([9]), 'nine'),
            (torch.ByteTensor([10]), 1010),
            (torch.ByteTensor([11]), 'eleven'),
            (torch.CharTensor([12]), 1212),
            (torch.CharTensor([13]), 'thirteen'),
            (torch.ShortTensor([14]), 1414),
            (torch.ShortTensor([15]), 'fifteen'),
            (torch.IntTensor([16]), 1616),
            (torch.IntTensor([17]), 'seventeen'),
            (torch.LongTensor([18]), 1818),
            (torch.LongTensor([19]), 'nineteen'),
        ]
        for obj, id in objects:
            local._objects[id]=obj
        test_json_dumps = [ # JSON dumps of the objects
            {'__FloatTensor__': '_fl.1111'},
            {'__FloatTensor__': '_fl.this_is_a_string_id'},
            {'__Variable__': '_fl.2222'},
            {'__Variable__': '_fl.three'},
            {'__Parameter__': '_fl.4444'},
            {'__Parameter__': '_fl.five'},
            {'__DoubleTensor__': '_fl.6666'},
            {'__DoubleTensor__': '_fl.seven'},
            {'__HalfTensor__': '_fl.8888'},
            {'__HalfTensor__': '_fl.nine'},
            {'__ByteTensor__': '_fl.1010'},
            {'__ByteTensor__': '_fl.eleven'},
            {'__CharTensor__': '_fl.1212'},
            {'__CharTensor__': '_fl.thirteen'},
            {'__ShortTensor__': '_fl.1414'},
            {'__ShortTensor__': '_fl.fifteen'},
            {'__IntTensor__': '_fl.1616'},
            {'__IntTensor__': '_fl.seventeen'},
            {'__LongTensor__': '_fl.1818'},
            {'__LongTensor__': '_fl.nineteen'}
            #{'__tuple__':}, TODO
            #{'__set__':}, TODO
            #{'__bytearray__':},TODO
            #{'__range__': }, TODO
            #{'__slice__': }  TODO
        ]
        json_decoder = PythonJSONDecoder(local)
        for i in range(len(test_json_dumps)):
            if i<= 19:
                self.assertIs(json_decoder.custom_obj_hook(test_json_dumps[i]), objects[i][0])
            else:
                self.assertEqual(json_decoder.custom_obj_hook(test_json_dumps[i]), objects[i][0])

