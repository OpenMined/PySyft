from unittest import TestCase
import syft as sy

class TestBaseWorker(TestCase):

    def test_search_obj(self):

        hook = sy.TorchHook()

        hook.local_worker.is_client_worker = False

        x = sy.Var(sy.FloatTensor([-2, -1, 0, 1, 2, 3])).set_id('#boston_housing #target #dataset')
        y = sy.Var(sy.FloatTensor([-2, -1, 0, 1, 2, 3])).set_id('#boston_housing #input #dataset')

        hook.local_worker.is_client_worker = True

        assert len(hook.local_worker.search("#boston_housing")) == 2
        assert len(hook.local_worker.search(["#boston_housing", "#target"])) == 1
