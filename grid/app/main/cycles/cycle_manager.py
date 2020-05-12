# Cycle module imports
from .cycle import Cycle
from .worker_cycle import WorkerCycle

# PyGrid modules
from ..storage.warehouse import Warehouse
from ..exceptions import CycleNotFoundError
from ..tasks.cycle import complete_cycle, run_task_once
from ..models import model_manager
from ..processes import process_manager

# Generic imports
from datetime import datetime, timedelta
from functools import reduce

import torch as th
import json
import logging
import random


class CycleManager:
    def __init__(self):
        self._cycles = Warehouse(Cycle)
        self._worker_cycles = Warehouse(WorkerCycle)

    def create(self, fl_process_id: str, version: str, cycle_time: int = 2500):
        """ Create a new federated learning cycle.
            Args:
                fl_process_id: FL Process's ID.
                version: Version (?)
                cycle_time: Remaining time to finish this cycle.
            Returns:
                fd_cycle: Cycle Instance.
        """
        _new_cycle = None

        # Retrieve a list of cycles using the same model_id/version
        sequence_number = len(
            self._cycles.query(fl_process_id=fl_process_id, version=version)
        )
        _now = datetime.now()
        _end = _now + timedelta(seconds=cycle_time)
        _new_cycle = self._cycles.register(
            start=_now,
            end=_end,
            sequence=sequence_number + 1,
            version=version,
            fl_process_id=fl_process_id,
        )

        return _new_cycle

    def last_participation(self, process: int, worker_id: str):
        """ Retrieve the last time the worker participated from this cycle.
            Args:
                process_id : Federated Learning Process ID.
                worker_id: Worker's ID.
            Returns:
                last_participation: last cycle.
        """
        _cycles = self._cycles.query(fl_process_id=process.id)

        last = 0
        if not len(_cycles):
            return last

        for cycle in _cycles:
            worker_cycle = self._worker_cycles.first(
                cycle_id=cycle.id, worker_id=worker_id
            )
            if worker_cycle and cycle.sequence > last:
                last = cycle.sequence

        return last

    def last(self, fl_process_id: int, version: str = None):
        """ Retrieve the last not completed registered cycle.
            Args:
                model_id: Model's ID.
                version: Model's version.
            Returns:
                cycle: Cycle Instance / None
        """
        if version:
            _cycle = self._cycles.last(
                fl_process_id=fl_process_id, version=version, is_completed=False
            )
        else:
            _cycle = self._cycles.last(fl_process_id=fl_process_id, is_completed=False)

        if not _cycle:
            raise CycleNotFoundError

        return _cycle

    def delete(self, **kwargs):
        """ Delete a registered Cycle.
            Args:
                model_id: Model's ID.
        """
        self._cycles.delete(**kwargs)

    def is_assigned(self, worker_id: str, cycle_id: str):
        """ Check if a workers is already assigned to an specific cycle.
            Args:
                worker_id : Worker's ID.
                cycle_id : Cycle's ID.
            Returns:
                result : Boolean Flag.
        """
        return self._worker_cycles.first(worker_id=worker_id, cycle_id=cycle_id) != None

    def assign(self, worker, cycle, hash_key: str):
        _worker_cycle = self._worker_cycles.register(
            worker=worker, cycle=cycle, request_key=hash_key
        )

        return _worker_cycle

    def validate(self, worker_id: str, cycle_id: str, request_key: str):
        """ Validate Worker's request key.
            Args:
                worker_id: Worker's ID.
                cycle_id: Cycle's ID.
                request_key: Worker's request key.
            Returns:
                result: Boolean flag
            Raises:
                CycleNotFoundError (PyGridError) : If not found any relation between the worker and cycle.
        """
        _worker_cycle = self._worker_cycles.first(
            worker_id=worker_id, cycle_id=cycle_id
        )

        if not _worker_cycle:
            raise CycleNotFoundError

        return _worker_cycle.request_key == request_key

    def count(self, **kwargs):
        return self._cycles.count(**kwargs)

    def submit_worker_diff(self, worker_id: str, request_key: str, diff: bin):
        """Submit reported diff
           Args:
                worker_id: Worker's ID.
                request_key: request (token) used by this worker during this cycle.
                diff: Model params trained by this worker.
           Returns:
                cycle_id : Cycle's ID.
           Raises:
                ProcessLookupError : If Not found any relation between the worker/cycle.
        """
        _worker_cycle = self._worker_cycles.first(
            worker_id=worker_id, request_key=request_key
        )

        if not _worker_cycle:
            raise ProcessLookupError

        _worker_cycle.is_completed = True
        _worker_cycle.completed_at = datetime.utcnow()
        _worker_cycle.diff = diff
        self._worker_cycles.update()

        # Run cycle end task async to we don't block report request
        # (for prod we probably should be replace this with Redis queue + separate worker)
        run_task_once("complete_cycle", complete_cycle, self, _worker_cycle.cycle_id)

    def complete_cycle(self, cycle_id: str):
        """Checks if the cycle is completed and runs plan avg"""
        logging.info("running complete_cycle for cycle_id: %s" % cycle_id)
        cycle = self._cycles.first(id=cycle_id)
        logging.info("found cycle: %s" % str(cycle))

        if cycle.is_completed:
            logging.info("cycle is already completed!")
            return

        server_config, _ = process_manager.get_configs(id=cycle.fl_process_id)
        logging.info("server_config: %s" % json.dumps(server_config, indent=2))
        completed_cycles_num = self._worker_cycles.count(
            cycle_id=cycle_id, is_completed=True
        )
        logging.info("# of diffs: %d" % completed_cycles_num)

        min_worker = server_config.get("min_worker", 3)
        max_worker = server_config.get("max_worker", 3)
        received_diffs_exceeds_min_worker = completed_cycles_num >= min_worker
        received_diffs_exceeds_max_worker = completed_cycles_num >= max_worker
        cycle_ended = True  # check cycle.cycle_time (but we should probably track cycle startime too)

        # Hmm, I don't think there should be such connection between ready_to_average, max_workers, and received_diffs
        # I thought max_workers just caps total number of simultaneous workers
        # 'cycle end' condition should probably depend on cycle_length regardless of number of actual received diffs
        # another 'cycle end' condition can be based on min_diffs
        ready_to_average = (
            True
            if (
                (received_diffs_exceeds_max_worker or cycle_ended)
                and received_diffs_exceeds_min_worker
            )
            else False
        )

        no_protocol = True  # only deal with plans for now

        logging.info("ready_to_average: %d" % int(ready_to_average))

        if ready_to_average and no_protocol:
            self._average_plan_diffs(server_config, cycle)

    def _average_plan_diffs(self, server_config: dict, cycle):
        """ skeleton code
                Plan only
                - get cycle
                - track how many has reported successfully
                - get diffs: list of (worker_id, diff_from_this_worker) on cycle._diffs
                - check if we have enough diffs? vs. max_worker
                - if enough diffs => average every param (by turning tensors into python matrices => reduce th.add => torch.div by number of diffs)
                - save as new model value => M_prime (save params new values)
                - create new cycle & new checkpoint
                at this point new workers can join because a cycle for a model exists
        """
        logging.info("start diffs averaging!")
        logging.info("cycle: %s" % str(cycle))
        logging.info("fl id: %d" % cycle.fl_process_id)
        _model = model_manager.get(fl_process_id=cycle.fl_process_id)
        logging.info("model: %s" % str(_model))
        model_id = _model.id
        logging.info("model id: %d" % model_id)
        _checkpoint = model_manager.load(model_id=model_id)
        logging.info("current checkpoint: %s" % str(_checkpoint))
        model_params = model_manager.unserialize_model_params(_checkpoint.values)
        logging.info("model params shapes: %s" % str([p.shape for p in model_params]))

        # Here comes simple hardcoded avg plan
        # it won't be always possible to retrieve and unserialize all diffs due to memory constrains
        # needs some kind of iterative or streaming approach,
        # e.g.
        # for diff_N in diffs:
        #    avg = avg_plan(avg, N, diff_N)
        # and the plan is:
        # avg_next = (avg_current*(N-1) + diff_N) / N
        reports_to_average = self._worker_cycles.query(
            cycle_id=cycle.id, is_completed=True
        )
        diffs = [
            model_manager.unserialize_model_params(report.diff)
            for report in reports_to_average
        ]

        # Again, not sure max_workers == number of diffs to avg
        diffs = random.sample(diffs, server_config.get("max_workers"))

        raw_diffs = [
            [diff[model_param] for diff in diffs]
            for model_param in range(len(model_params))
        ]
        logging.info("raw diffs lengths: %s" % str([len(row) for row in raw_diffs]))

        sums = [reduce(th.add, param) for param in raw_diffs]
        logging.info("sums shapes: %s" % str([sum.shape for sum in sums]))

        diff_avg = [th.div(param, len(diffs)) for param in sums]
        logging.info("diff_avg shapes: %s" % str([d.shape for d in diff_avg]))

        # apply avg diff!
        _updated_model_params = [
            model_param - diff_param
            for model_param, diff_param in zip(model_params, diff_avg)
        ]
        logging.info(
            "_updated_model_params shapes: %s"
            % str([p.shape for p in _updated_model_params])
        )

        # make new checkpoint
        serialized_params = model_manager.serialize_model_params(_updated_model_params)
        _new_checkpoint = model_manager.save(model_id, serialized_params)
        logging.info("new checkpoint: %s" % str(_new_checkpoint))

        # mark current cycle completed
        cycle.is_completed = True
        self._cycles.update()

        completed_cycles_num = self._cycles.count(
            fl_process_id=cycle.fl_process_id, is_completed=True
        )
        logging.info("completed_cycles_num: %d" % completed_cycles_num)
        max_cycles = server_config.get("num_cycles")
        if completed_cycles_num < max_cycles:
            # make new cycle
            _new_cycle = self.create(cycle.fl_process_id, cycle.version)
            logging.info("new cycle: %s" % str(_new_cycle))
        else:
            logging.info("FL is done!")
