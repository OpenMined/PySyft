# Syft assets module imports
from .plan import Plan

# PyGrid imports
from ..storage.warehouse import Warehouse
from ..exceptions import PlanNotFoundError


class PlanManager:
    def __init__(self):
        self._plans = Warehouse(Plan)

    def register(self, process, plans: dict, avg_plan: bool):
        if not avg_plan:
            # Register new Plans into the database
            for key, value in plans.items():
                self._plans.register(name=key, value=value, plan_flprocess=process)
        else:
            # Register the average plan into the database
            self._plans.register(value=plans, avg_flprocess=process, is_avg_plan=True)

    def get(self, **kwargs):
        """ Retrieve the desired plans.
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
        """ Retrieve the first occurence that matches with query.
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
        """ Delete a registered Plan.
            Args:
                query: Query used to identify the plan object.
        """
        self._plans.delete(**kwargs)
