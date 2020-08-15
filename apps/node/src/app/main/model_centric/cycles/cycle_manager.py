# Cycle module imports
import json
import logging

# Generic imports
from datetime import datetime, timedelta
from functools import reduce

import torch as th

from ...core.exceptions import CycleNotFoundError

# PyGrid modules
from ...core.warehouse import Warehouse
from ..models import model_manager
from ..processes import process_manager
from ..syft_assets import PlanManager
from ..tasks.cycle import complete_cycle, run_task_once
from .cycle import Cycle
from .worker_cycle import WorkerCycle


class CycleManager:
    def __init__(self):
        self._cycles = Warehouse(Cycle)
        self._worker_cycles = Warehouse(WorkerCycle)

    def create(self, fl_process_id: int, version: str, cycle_time: int):
        """Create a new federated learning cycle.

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
        _end = _now + timedelta(seconds=cycle_time) if cycle_time is not None else None
        _new_cycle = self._cycles.register(
            start=_now,
            end=_end,
            sequence=sequence_number + 1,
            version=version,
            fl_process_id=fl_process_id,
        )

        return _new_cycle

    def last_participation(self, process: int, worker_id: str):
        """Retrieve the last time the worker participated from this cycle.

        Args:
            process: Federated Learning Process.
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
        """Retrieve the last not completed registered cycle.

        Args:
            fl_process_id: Federated Learning Process ID.
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
        """Delete a registered Cycle.

        Args:
            model_id: Model's ID.
        """
        self._cycles.delete(**kwargs)

    def is_assigned(self, worker_id: str, cycle_id: int):
        """Check if a workers is already assigned to an specific cycle.

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

    def validate(self, worker_id: str, cycle_id: int, request_key: str):
        """Validate Worker's request key.

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

        logging.info(f"Updating worker cycle: {str(_worker_cycle)}")

        _worker_cycle.is_completed = True
        _worker_cycle.completed_at = datetime.utcnow()
        _worker_cycle.diff = diff
        self._worker_cycles.update()

        # Run cycle end task async to we don't block report request
        # (for prod we probably should be replace this with Redis queue + separate worker)
        run_task_once("complete_cycle", complete_cycle, self, _worker_cycle.cycle_id)

    def complete_cycle(self, cycle_id: int):
        """Checks if the cycle is completed and runs plan avg."""
        logging.info("running complete_cycle for cycle_id: %s" % cycle_id)
        cycle = self._cycles.first(id=cycle_id)
        logging.info("found cycle: %s" % str(cycle))

        if cycle.is_completed:
            logging.info("cycle is already completed!")
            return

        server_config, _ = process_manager.get_configs(id=cycle.fl_process_id)
        logging.info("server_config: %s" % json.dumps(server_config, indent=2))

        received_diffs = self._worker_cycles.count(cycle_id=cycle_id, is_completed=True)
        logging.info("# of diffs: %d" % received_diffs)

        min_diffs = server_config.get("min_diffs", None)
        max_diffs = server_config.get("max_diffs", None)

        hit_diffs_limit = (
            received_diffs >= max_diffs if max_diffs is not None else False
        )
        hit_time_limit = datetime.now() >= cycle.end if cycle.end is not None else False
        no_limits = max_diffs is None and cycle.end is None
        has_enough_diffs = (
            received_diffs >= min_diffs if min_diffs is not None else True
        )

        ready_to_average = has_enough_diffs and (
            no_limits or hit_diffs_limit or hit_time_limit
        )

        no_protocol = True  # only deal with plans for now

        logging.info("ready_to_average: %d" % int(ready_to_average))

        if ready_to_average and no_protocol:
            self._average_plan_diffs(server_config, cycle)

    def _average_plan_diffs(self, server_config: dict, cycle):
        """skeleton code Plan only.

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
        model_params = model_manager.unserialize_model_params(_checkpoint.value)
        logging.info("model params shapes: %s" % str([p.shape for p in model_params]))

        reports_to_average = self._worker_cycles.query(
            cycle_id=cycle.id, is_completed=True
        )

        diffs = [
            model_manager.unserialize_model_params(report.diff)
            for report in reports_to_average
        ]

        avg_plan_rec = process_manager.get_plan(
            fl_process_id=cycle.fl_process_id, is_avg_plan=True
        )

        if avg_plan_rec and avg_plan_rec.value:
            logging.info("Doing hosted avg plan")
            avg_plan = PlanManager.deserialize_plan(avg_plan_rec.value)

            # check if the uploaded avg plan is iterative or not
            iterative_plan = server_config.get("iterative_plan", False)

            # diffs if list [diff1, diff2, ...] of len == received_diffs
            # each diff is list [param1, param2, ...] of len == model params
            # diff_avg is list [param1_avg, param2_avg, ...] of len == model params
            if iterative_plan:
                diff_avg = diffs[0]
                for i, diff in enumerate(diffs[1:]):
                    diff_avg = avg_plan(list(diff_avg), diff, th.tensor([i + 1]))
            else:
                diff_avg = avg_plan(diffs)

        else:
            # Fallback to simple hardcoded avg plan
            logging.info("Doing hardcoded avg plan")
            raw_diffs = [
                [diff[model_param] for diff in diffs]
                for model_param in range(len(model_params))
            ]
            # raw_diffs is [param1_diffs, param2_diffs, ...] with len == num of model params
            # each param1_diffs is [ diff1, diff2, ... ] with len == num of received_diffs
            # diff_avg going to be [ param1_diffs_avg, param2_diffs_avg, ...] with len == num of model params
            # where param1_diffs_avg is avg of tensors in param1_diffs
            logging.info("raw diffs lengths: %s" % str([len(row) for row in raw_diffs]))
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
        max_cycles = server_config.get("num_cycles", 0)
        if completed_cycles_num < max_cycles or max_cycles == 0:
            # make new cycle
            _new_cycle = self.create(
                cycle.fl_process_id, cycle.version, server_config.get("cycle_length")
            )
            logging.info("new cycle: %s" % str(_new_cycle))
        else:
            logging.info("FL is done!")
