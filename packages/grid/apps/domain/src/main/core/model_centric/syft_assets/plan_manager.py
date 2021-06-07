# third party
import syft as sy
from syft import deserialize
from syft import serialize
from syft.proto.core.plan.plan_pb2 import Plan as PlanPB

# grid relative
from ...exceptions import PlanInvalidError
from ...exceptions import PlanNotFoundError
from ...exceptions import PlanTranslationError
from ...manager.database_manager import DatabaseManager
from .plan import Plan


class PlanManager(DatabaseManager):
    schema = Plan

    def __init__(self, database):
        self._schema = PlanManager.schema
        self.db = database

    def register(self, process, plans: dict, avg_plan: bool):
        if not avg_plan:
            # Store specific plan types in proper fields in DB
            plans_types = {}
            for idx_type, plan in plans.items():
                type = "syft"
                idx = idx_type
                if ":" in idx_type:
                    idx, type = idx_type.split(":", 2)
                if idx not in plans_types:
                    plans_types[idx] = {}
                plans_types[idx][type] = plan

            # Register new Plans into the database
            for key, plans in plans_types.items():
                super().register(
                    name=key,
                    value=plans.get("syft", None),
                    value_ts=plans.get("ts", None),
                    value_tfjs=plans.get("tfjs", None),
                    plan_flprocess=process,
                )
        else:
            # Register the average plan into the database
            super().register(value=plans, avg_flprocess=process, is_avg_plan=True)

    def get(self, **kwargs):
        """Retrieve the desired plans.

        Args:
            query : query used to identify the desired plans object.
        Returns:
            plan : Plan list or None if it wasn't found.
        Raises:
            PlanNotFound (PyGridError) : If Plan not found.
        """
        _plans = self.query(**kwargs)

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
        _plan = super().first(**kwargs)

        if not _plan:
            raise PlanNotFoundError

        return _plan

    def delete(self, **kwargs):
        """Delete a registered Plan.

        Args:
            query: Query used to identify the plan object.
        """
        super().delete(**kwargs)

    @staticmethod
    def deserialize_plan(bin: bytes) -> "sy.Plan":
        """Deserialize a Plan."""
        pb = PlanPB()
        pb.ParseFromString(bin)
        plan = deserialize(pb)
        return plan

    @staticmethod
    def serialize_plan(plan: "sy.Plan") -> bin:
        """Serialize a Plan."""
        pb = serialize(plan)
        serialized_plan = pb.SerializeToString()
        return serialized_plan
