import uuid


class Scope:
    """ An abstraction to private rooms with some workers that will perform a plan/protocol process. """

    def __init__(self, worker_id: str, protocol):
        """ Create a scope instance.
            
            Args:
                worker_id : Id of the worker that creates this scope.
                protocol : Syft plans/protocol structure.
        """
        self.creator = worker_id
        self.id = str(uuid.uuid4())
        self.assignments = {self.creator: "assignment-1"}
        self.plan_number = {self.creator: 0}
        self.protocol = protocol

    def add_participant(self, participant_id: str):
        """ Add a new participant on this scope.
            
            Args:
                participant_id : worker id of the new participant.
        """
        # Check if already exist an assignment / plan number defined to this id.
        assignment = self.assignments.get(participant_id, None)
        plan_number = self.plan_number.get(participant_id, None)

        if not assignment:
            self.assignments[participant_id] = "assignment-" + str(
                len(self.assignments) + 1
            )

        if participant_id not in self.plan_number:
            self.plan_number[participant_id] = len(self.plan_number)

    def get_role(self, participant_id: str) -> str:
        """ Get participant's current role.
            
            Args:
                participant_id: Worker's id.
            Returns:
                role : Participant's role.
        """
        if participant_id in self.assignments:
            if participant_id == self.creator:
                return "creator"
            else:
                return "participant"

    def get_plan(self, participant_id: str) -> int:
        """ Get participant's plan index.

            Args:
                participant_id: Worker's Id.
            Returns:
                index : Participant's plan index.
        """
        return self.plan_number[participant_id]

    def get_assignment(self, participant_id: str) -> str:
        """ Get participant's assignment.
            
            Args:
                participant_id : Worker's Id.
            Returns:
                assignment : Participant's assignment.
        """
        return self.assignments[participant_id]

    def get_participants(self) -> list:
        """ Get a list of participants' id of this scope.
            
            Returns:
                participants : List of scope's participants.
        """
        return filter(lambda x: self.plan_number[x] != 0, self.plan_number.keys())
