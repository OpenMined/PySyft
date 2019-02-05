class FederatedDataset:
    def __init__(self, inputs: dict, targets: dict):
        """This class takes two dictionaries of datapoints and forms
        them into batches ready for training. The keys to both inputs
        and target are worker IDs, and the values are tensors. The class
        will use these collections of pointers to datasets and form them
        into one cohesive dataset (with some statistics on how many data-
        points had to be thrown away, if any).

        Args:
            inputs (dict): dict of dataset tensors or pointers to dataset tensors
            targets (dict): dict of target/label tensors or pointers to target/label tensors
            batch_size (int): size of each batch
            num_iterators (int): number of iterators to use for parallel computations
        """

        # first check to see which people line up
        input_workers = set(inputs.keys())
        target_workers = set(targets.keys())
        intersection = input_workers.intersection(target_workers)
        diff = input_workers.difference(target_workers)

        if len(diff) > 0:
            print("Data from some workers ignored:" + str(diff))

        self.workers = list(intersection)

        self.worker2num_rows = {}
        self.worker2inputs = {}
        self.worker2targets = {}

        self.num_rows = 0

        for worker in self.workers:

            if (len(inputs[worker]) != 1) or (len(targets[worker]) != 1):
                raise Exception(
                    "There must be only one input and one target"
                    + "tensor per worker. Otherwise it's not clear"
                    + "which rows in the input correspond to which"
                    + "rows in the output"
                )

            worker_inputs = inputs[worker][0]
            worker_targets = targets[worker][0]

            if worker_inputs.shape[0] != worker_targets.shape[0]:
                raise Exception(
                    "On each worker, the input and target must have" + "the same number of rows."
                )

            self.worker2num_rows[worker] = worker_inputs.shape[0]
            self.worker2inputs[worker] = worker_inputs
            self.worker2targets[worker] = worker_targets

            self.num_rows += worker_inputs.shape[0]

    def __len__(self):
        return self.num_rows
