class FederatedDataset:
    def __init__(self, inputs: dict, targets: dict, batch_size=12, num_iterators=1):
        """This class takes two dictionaries of datapoints and forms
        them into batches ready for training. The keys to both inputs
        and target are worker IDs, and the values are tensors. The class
        will use these collections of pointers to datasets and form them
        into one cohesive dataset (with some statistics on how many data-
        points had to be thrown away, if any)."""

        # first check to see which people line up
        input_workers = set(inputs.keys())
        target_workers = set(targets.keys())
        intersection = input_workers.intersection(target_workers)
        diff = input_workers.difference(target_workers)

        if len(diff) == 0:
            ""
        else:
            print("Data from some workers ignored:" + str(diff))

        self.workers = list(intersection)

        self.batch_size = batch_size
        # You can't have more iterators than workers
        self.num_iterators = min(num_iterators, len(self.workers) - 1)

        self.worker2num_rows = {}
        self.worker2inputs = {}
        self.worker2targets = {}

        self.row_i = 0
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

        self.iterators = list()
        for i in range(self.num_iterators):
            self.iterators.append(FederatedIterator(self, self.worker2num_rows, self.workers))

        self.reset()

    def reset(self):
        self.row_i = 0
        for i, iterator in enumerate(self.iterators):
            iterator.reset(i)

    def keep_going(self):
        if self.row_i < self.num_rows:
            return True
        return False

    def step(self):

        if self.num_iterators > 1:
            batches = {}
            for i in range(self.num_iterators):
                data, target = self.iterators[i].step()
                batches[data.location] = (data, target)
                self.row_i += data.shape[0]
            return batches
        else:
            data, target = self.iterators[0].step()
            self.row_i += data.shape[0]
            return (data, target)


class FederatedIterator:
    def __init__(self, federated_dataset, worker2num_rows, workers):

        self.worker2num_rows = worker2num_rows
        self.workers = workers
        self.fd = federated_dataset

    def step(self):
        worker = self.workers[self.worker_iterator]
        wi = self.worker2iterators[worker]
        input_batch = self.fd.worker2inputs[worker][wi : wi + self.fd.batch_size]
        target_batch = self.fd.worker2targets[worker][wi : wi + self.fd.batch_size]

        self.worker2iterators[worker] += self.fd.batch_size
        if self.worker2iterators[worker] >= self.worker2num_rows[worker]:
            self.worker2iterators[worker] = 0
            new_iterator = (self.worker_iterator + 1) % len(self.workers)

            keepgoing = True
            while keepgoing:
                keepgoing = False
                for i, it in enumerate(self.fd.iterators):
                    if it.worker_iterator == new_iterator:
                        keepgoing = True
                        new_iterator = (new_iterator + 1) % len(self.workers)
                        break

            self.worker_iterator = new_iterator

        return (input_batch, target_batch)

    def reset(self, worker_iterator):
        self.worker_iterator = worker_iterator
        self.worker2iterators = {}
        for worker in self.workers:
            self.worker2iterators[worker] = 0
