from collections import defaultdict
from typing import List, Tuple, Union

import syft as sy
from syft.execution.plan import func2plan
from syft.workers.base import BaseWorker




class PrimitiveStorage:
    """
    Used by normal workers to store crypto primitives
    Used by crypto providers to build crypto primitives
    """
    def __init__(self, owner):
        self.fss_eq = []
        self.fss_comp = []
        self.beaver = []

        self._owner: BaseWorker = owner
        self._builders = {
            "fss_eq": self.get_fss_plan(type_op="eq"),
            "fss_comp": self.get_fss_plan(type_op="comp"),
            "beaver": self.build_triples
        }

    def provide_primitives(self, type:str, workers: List[BaseWorker], n_instances: int=10, **kwargs):
        builder = self._builders[type]

        primitives = []
        for i in range(n_instances):
            primitive_instance: Tuple[Tuple] = builder(n_party=len(workers), **kwargs)
            primitives.append(primitive_instance)

        for i, worker in enumerate(workers):
            worker_primitives = [primitive[i] for primitive in primitives]

            self._owner.send_command()








    def get_fss_plan(self, type_op):
        if type_op == "eq":
            fss_class = sy.frameworks.torch.mpc.fss.DPF
        elif type_op == "comp":
            fss_class = sy.frameworks.torch.mpc.fss.DIF
        else:
            raise ValueError(f"type_op {type_op} not valid")

        return fss_class.keygen


    def build_triples(self, **kwargs):
        pass

