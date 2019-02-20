class FederatedData:
    """
    A piece of data which is distributed across workers
    """

    def __init__(self, data_dict):
        self.worker_ids = set()
        for worker_id, worker_data in data_dict.items():
            # Allow worker_data to be a list of size 1
            if isinstance(worker_data, list):
                if len(worker_data) == 1:
                    worker_data = worker_data[0]
                else:
                    raise ReferenceError(
                        f"There should be exactly one dataset provided per worker, found{len(worker_data)}."
                    )

            self.worker_ids.add(worker_id)
            setattr(self, worker_id, worker_data)

    def __getitem__(self, indices):
        if isinstance(indices, tuple):
            worker_id, item_id = indices
            return getattr(self, worker_id)[item_id]
        else:
            worker_id = indices
            return getattr(self, worker_id)

    def __len__(self):
        data_length = 0
        for worker_id in self.worker_ids:
            data_length += self[worker_id].shape[0]
        return data_length

    @property
    def workers(self):
        return self.worker_ids

    def drop_worker(self, worker_id):
        if worker_id in self.worker_ids:
            delattr(self, worker_id)
            self.worker_ids.remove(worker_id)

    def __repr__(self):
        fmt_str = "FederatedData\n"
        fmt_str += "    Distributed accross: {}\n".format(", ".join(self.workers))
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str


class FederatedDataset:
    def __init__(self, federated_inputs, federated_targets):
        """This class takes two dictionaries of data points, the keys
        of which are worker IDs, and their values are tensors or pointers
        to tensors. One can also provide direct FederatedData elements.
        The class will use these collections of datasets or pointers
        to datasets and form them into one cohesive dataset.
        Args:
            federated_inputs (dict or FederatedData): dict of dataset tensors
            or pointers to dataset tensors
            federated_targets (dict or FederatedData): dict of target/label
            tensors or pointers to target/label tensors
        """
        # Convert in the Federated format if needed
        if isinstance(federated_inputs, dict):
            federated_inputs = FederatedData(federated_inputs)
        self.data = federated_inputs
        if isinstance(federated_targets, dict):
            federated_targets = FederatedData(federated_targets)
        self.targets = federated_targets

        # Check to see which people line up
        input_workers = federated_inputs.workers
        target_workers = federated_targets.workers
        skewed_workers = input_workers.difference(target_workers)

        # Keep only workers that appear in both the data and the targets
        if len(skewed_workers) > 0:
            print("Data from some workers ignored:", skewed_workers)
            for worker_id in skewed_workers:
                self.data.drop_worker(worker_id)
                self.targets.drop_worker(worker_id)

        # Check that data and targets for a worker are consistent
        for worker in self.workers:
            assert len(self.data[worker]) == len(self.targets[worker]), (
                "On each worker, the input and target must have" + "the same number of rows."
            )

    @property
    def workers(self):
        return self.data.workers

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        fmt_str = "FederatedDataset\n"
        fmt_str += "    Distributed accross: {}\n".format(", ".join(self.workers))
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        return fmt_str
