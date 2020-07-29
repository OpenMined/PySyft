# Syft assets module imports
# Syft dependencies
import syft as sy
from syft.execution.translation.default import PlanTranslatorDefault
from syft.execution.translation.threepio import PlanTranslatorTfjs
from syft.execution.translation.torchscript import PlanTranslatorTorchscript
from syft.serde import protobuf
from syft_proto.execution.v1.plan_pb2 import Plan as PlanPB

from ...core.exceptions import PlanInvalidError, PlanNotFoundError, PlanTranslationError

# PyGrid imports
from ...storage.warehouse import Warehouse
from .plan import Plan

# Make fake local worker for serialization
worker = sy.VirtualWorker(hook=None)


class PlanManager:
    def __init__(self):
        self._plans = Warehouse(Plan)

    def register(self, process, plans: dict, avg_plan: bool):
        if not avg_plan:
            # Convert client plans to specific formats
            plans_converted = {}
            for idx, plan_ser in plans.items():
                try:
                    plan = self.deserialize_plan(plan_ser)
                except:
                    raise PlanInvalidError()
                try:
                    plan_ops = self.trim_plan(plan, "default")
                    plan_ts = self.trim_plan(plan, "torchscript")
                    plan_tfjs = self.trim_plan(plan, "tfjs")
                    plan_ops_ser = self.serialize_plan(plan_ops)
                    plan_ts_ser = self.serialize_plan(plan_ts)
                    plan_tfjs_ser = self.serialize_plan(plan_tfjs)
                except:
                    raise PlanTranslationError()
                plans_converted[idx] = {
                    "list": plan_ops_ser,
                    "torchscript": plan_ts_ser,
                    "tfjs": plan_tfjs_ser,
                }

            # Register new Plans into the database
            for key, plan in plans_converted.items():
                self._plans.register(
                    name=key,
                    value=plan["list"],
                    value_ts=plan["torchscript"],
                    value_tfjs=plan["tfjs"],
                    plan_flprocess=process,
                )
        else:
            # Register the average plan into the database
            self._plans.register(value=plans, avg_flprocess=process, is_avg_plan=True)

    def get(self, **kwargs):
        """Retrieve the desired plans.

        Args:
            query : query used to identify the desired plans object.
        Returns:
            plan : Plan list or None if it wasn't found.
        Raises:
            PlanNotFound (PyGridError) : If Plan not found.
        """
        _plans = self._plans.query(**kwargs)

        if not _plans:
            raise PlanNotFoundError

        return _plans

    def first(self, **kwargs):
        """Retrieve the first occurrence that matches with query.

        Args:
            query : query used to identify the desired plans object.
        Returns:
            plan : Plan Instance or None if it wasn't found.
        Raises:
            PlanNotFound (PyGridError) : If Plan not found.
        """
        _plan = self._plans.first(**kwargs)

        if not _plan:
            raise PlanNotFoundError

        return _plan

    def delete(self, **kwargs):
        """Delete a registered Plan.

        Args:
            query: Query used to identify the plan object.
        """
        self._plans.delete(**kwargs)

    @staticmethod
    def deserialize_plan(bin: bin) -> "sy.Plan":
        """Deserialize a Plan."""
        pb = PlanPB()
        pb.ParseFromString(bin)
        plan = protobuf.serde._unbufferize(worker, pb)
        return plan

    @staticmethod
    def serialize_plan(plan: "sy.Plan") -> bin:
        """Serialize a Plan."""
        pb = protobuf.serde._bufferize(worker, plan)
        serialized_plan = pb.SerializeToString()
        return serialized_plan

    @staticmethod
    def trim_plan(plan: "sy.Plan", variant: str) -> "sy.Plan":
        """Trim Plan to specified variant."""

        type_translators = {
            "torchscript": PlanTranslatorTorchscript,
            "default": PlanTranslatorDefault,
        }

        fw_translators = {
            "tfjs": PlanTranslatorTfjs,
        }

        if variant not in type_translators and variant not in fw_translators:
            raise PlanTranslationError

        plan_copy = plan.copy()

        if variant in type_translators:
            for name, cls in type_translators.items():
                if name != variant:
                    plan_copy.remove_translation(cls)

        if variant in fw_translators:
            # First, leave only default translation
            for name, cls in type_translators.items():
                if name != "default":
                    plan_copy.remove_translation(cls)
            # Set actions to be specific type
            plan_copy.base_framework = variant
            # Remove other actions
            plan_copy.roles = None

        return plan_copy
