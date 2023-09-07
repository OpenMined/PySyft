# future
from __future__ import annotations

# stdlib
import copy
import hashlib
import textwrap
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

# third party
import pydantic
from pydantic import validator
from rich.progress import Progress
from typing_extensions import Self

# relative
from ...client.api import NodeIdentity
from ...client.client import SyftClient
from ...client.client import SyftClientSessionCache
from ...node.credentials import SyftSigningKey
from ...node.credentials import SyftVerifyKey
from ...serde.serializable import serializable
from ...serde.serialize import _serialize
from ...service.metadata.node_metadata import NodeMetadata
from ...store.linked_obj import LinkedObject
from ...types.datetime import DateTime
from ...types.identity import Identity
from ...types.identity import UserIdentity
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.syft_object import short_qual_name
from ...types.transforms import TransformContext
from ...types.transforms import rename
from ...types.transforms import transform
from ...types.uid import UID
from ...util import options
from ...util.colors import SURFACE
from ...util.markdown import markdown_as_class_with_fields
from ...util.util import full_name_with_qualname
from ..code.user_code import SubmitUserCode
from ..network.network_service import NodePeer
from ..network.routes import NodeRoute
from ..network.routes import connection_to_route
from ..request.request import Request
from ..request.request import RequestStatus
from ..response import SyftError
from ..response import SyftException
from ..response import SyftNotReady
from ..response import SyftSuccess


@serializable()
class EventAlreadyAddedException(SyftException):
    pass


@transform(NodeMetadata, NodeIdentity)
def metadata_to_node_identity() -> List[Callable]:
    return [rename("id", "node_id"), rename("name", "node_name")]


class ProjectEvent(SyftObject):
    __canonical_name__ = "ProjectEvent"
    __version__ = SYFT_OBJECT_VERSION_1

    __hash_exclude_attrs__ = ["event_hash", "signature"]

    # 1. Creation attrs
    id: UID
    timestamp: DateTime
    allowed_sub_types: Optional[List] = []
    # 2. Rebase attrs
    project_id: Optional[UID]
    seq_no: Optional[int]
    prev_event_uid: Optional[UID]
    prev_event_hash: Optional[str]
    event_hash: Optional[str]
    # 3. Signature attrs
    creator_verify_key: Optional[SyftVerifyKey]
    signature: Optional[bytes]  # dont use in signing

    def __repr_syft_nested__(self):
        return (
            short_qual_name(full_name_with_qualname(self)),
            f"{str(self.id)[:4]}...{str(self.id)[-3:]}",
        )

    @pydantic.root_validator(pre=True)
    def make_timestamp(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if "timestamp" not in values or values["timestamp"] is None:
            values["timestamp"] = DateTime.now()
        return values

    def _pre_add_update(self, project: Project) -> None:
        pass

    def rebase(self, project: Project) -> Self:
        prev_event = project.events[-1] if project.events else None
        self.project_id = project.id

        if prev_event:
            self.prev_event_uid = prev_event.id
            self.prev_event_hash = prev_event.event_hash
            self.seq_no = prev_event.seq_no + 1
        else:
            self.prev_event_uid = project.id
            self.prev_event_hash = project.start_hash
            self.seq_no = 1

        return self

    @property
    def valid(self) -> Union[SyftSuccess, SyftError]:
        if self.signature is None:
            return SyftError(message="Sign event first")
        try:
            # Recompute hash
            event_hash_bytes, current_hash = create_project_event_hash(self)
            if current_hash != self.event_hash:
                raise Exception(
                    f"Event hash {current_hash} does not match {self.event_hash}"
                )

            self.creator_verify_key.verify_key.verify(event_hash_bytes, self.signature)
            return SyftSuccess(message="Event signature is valid")
        except Exception as e:
            return SyftError(message=f"Failed to validate message. {e}")

    def valid_descendant(
        self, project: Project, prev_event: Optional[Self]
    ) -> Union[SyftSuccess, SyftError]:
        valid = self.valid
        if not valid:
            return valid

        if prev_event:
            prev_event_id = prev_event.id
            prev_event_hash = prev_event.event_hash
            prev_seq_no = prev_event.seq_no
        else:
            prev_event_id = project.id
            prev_event_hash = project.start_hash
            prev_seq_no = 0

        if self.prev_event_uid != prev_event_id:
            return SyftError(
                message=f"{self} prev_event_uid: {self.prev_event_uid} "
                "does not match {prev_event_id}"
            )

        if self.prev_event_hash != prev_event_hash:
            return SyftError(
                message=f"{self} prev_event_hash: {self.prev_event_hash} "
                "does not match {prev_event_hash}"
            )

        if self.seq_no != prev_seq_no + 1:
            return SyftError(
                message=f"{self} seq_no: {self.seq_no} "
                "is not subsequent to {prev_seq_no}"
            )

        if self.project_id != project.id:
            return SyftError(
                message=f"{self} project_id: {self.project_id} "
                "does not match {project.id}"
            )

        if hasattr(self, "parent_event_id"):
            parent_event = project.event_id_hashmap[self.parent_event_id]
            if type(self) not in parent_event.allowed_sub_types:
                return SyftError(
                    message=f"{self} is not a valid subevent" f"for {parent_event}"
                )
        return SyftSuccess(message=f"{self} is valid descendant of {prev_event}")

    def sign(self, signing_key: SyftSigningKey) -> None:
        if self.creator_verify_key != signing_key.verify_key:
            raise Exception(
                f"creator_verify_key has changed from: {self.creator_verify_key} to "
                f"{signing_key.verify_key}"
            )
        # Calculate Hash
        event_hash_bytes, event_hash = create_project_event_hash(self)
        self.event_hash = event_hash

        # Sign Hash
        signed_obj = signing_key.signing_key.sign(event_hash_bytes)
        self.signature = signed_obj._signature

    def publish(self, project: Project) -> Union[SyftSuccess, SyftError]:
        try:
            result = project.add_event(self)
            return result
        except EventAlreadyAddedException:  # nosec
            return SyftSuccess(message="Event already added")


class ProjectEventAddObject(ProjectEvent):
    __canonical_name__ = "ProjectEventAddObject"
    __version__ = SYFT_OBJECT_VERSION_1


class ProjectEventAddLink(ProjectEvent):
    __canonical_name__ = "ProjectEventAddLink"
    __version__ = SYFT_OBJECT_VERSION_1


# Project Sub Event are the events which tend to describe the main events
# For example, if a project event is created, then the project sub event will be the
# 1. ProjectThreadMessage
# 2. Emojis
# 3. Approval for request based events
# Mainly these events are used to describe the main events
# For example, we could make our messaging system to have only one level of thread messages
# it would also help us define the expected sub events for each event
# such that only allowed events could be the sub type of the main event
class ProjectSubEvent(ProjectEvent):
    __canonical_name__ = "ProjectSubEvent"
    __version__ = SYFT_OBJECT_VERSION_1

    parent_event_id: UID


@serializable()
class ProjectThreadMessage(ProjectSubEvent):
    __canonical_name__ = "ProjectThreadMessage"
    __version__ = SYFT_OBJECT_VERSION_1

    message: str


@serializable()
class ProjectMessage(ProjectEventAddObject):
    __canonical_name__ = "ProjectMessage"
    __version__ = SYFT_OBJECT_VERSION_1

    message: str
    allowed_sub_types: List[Type] = [ProjectThreadMessage]

    def reply(self, message: str) -> ProjectMessage:
        return ProjectThreadMessage(message=message, parent_event_id=self.id)


@serializable()
class ProjectRequestResponse(ProjectSubEvent):
    __canonical_name__ = "ProjectRequestResponse"
    __version__ = SYFT_OBJECT_VERSION_1

    response: bool


@serializable()
class ProjectRequest(ProjectEventAddObject):
    __canonical_name__ = "ProjectRequest"
    __version__ = SYFT_OBJECT_VERSION_1

    linked_request: LinkedObject
    allowed_sub_types: List[Type] = [ProjectRequestResponse]

    @validator("linked_request", pre=True)
    def _validate_linked_request(cls, v):
        if isinstance(v, Request):
            linked_request = LinkedObject.from_obj(v, node_uid=v.node_uid)
            return linked_request
        elif isinstance(v, LinkedObject):
            return v
        else:
            raise ValueError(
                f"linked_request should be either Request or LinkedObject, got {type(v)}"
            )

    @property
    def request(self):
        return self.linked_request.resolve

    __repr_attrs__ = [
        "request.status",
        "request.changes[-1].link.service_func_name",
    ]

    def _repr_markdown_(self) -> str:
        func_name = None
        if len(self.request.changes) > 0:
            func_name = self.request.changes[-1].link.service_func_name
        repr_dict = {
            "request.status": self.request.status,
            "request.changes[-1].link.service_func_name": func_name,
        }
        return markdown_as_class_with_fields(self, repr_dict)

    def approve(self) -> ProjectRequestResponse:
        result = self.request.approve()
        if isinstance(result, SyftError):
            return result
        return ProjectRequestResponse(response=True, parent_event_id=self.id)

    def accept_by_depositing_result(self, result: Any, force: bool = False):
        return self.request.accept_by_depositing_result(result=result, force=force)

    # TODO: To add deny requests, when deny functionality is added

    def status(self, project: Project) -> Union[Dict, SyftError]:
        """Returns the status of the request

        Args:
            project (Project): Project object to check the status

        Returns:
            str: Status of the request

        During Request  status calculation, we do not allow multiple responses
        """
        responses = project.get_children(self)
        if len(responses) == 0:
            return "No one has responded to the request yet. Kindly recheck later 🙂"

        if len(responses) > 1:
            return SyftError(
                message="The Request Contains more than one Response"
                "which is currently not possible"
                "The request should contain only one response"
                "Kindly re-submit a new request"
                "The Syft Team is working on this issue to handle multiple responses"
            )
        response = responses[0]
        if not isinstance(response, ProjectRequestResponse):
            return SyftError(
                message=f"Response : {type(response)} is not of type ProjectRequestResponse"
            )

        print("Request Status : ", "Approved" if response.response else "Denied")


def poll_creation_wizard() -> List[Any]:
    w = textwrap.TextWrapper(initial_indent="\t", subsequent_indent="\t")

    welcome_msg = "Welcome to the Poll Creation Wizard 🧙‍♂️ 🪄!!!"

    description1 = """You've arrived here because you were interested in a creating poll.
A poll is a way to gather opinions or information from a group of people.
It typically involves asking a specific question and providing a set of answer choices for respondents to choose from"""

    description2 = """In a poll, the question is the inquiry being asked of the participants.
It should be a multiple-choice. The choices are the options that respondents
are given to select as their answer to the question.For example, a poll question might be
"Which is your favorite color?" with answer choices of red, blue, green, and yellow.
Participants can then select the answer that best represents their opinion or preference"""

    description3 = """Since you didn't pass in questions, choices into .create_poll() (or you did so incorrectly),
this wizard is going to guide you through the process of creating a poll."""

    description4 = """In this wizard, we're going to ask you for a question and list of choices
to create the poll. The Questions and choices are converted to strings"""

    print("\t" + "=" * 69)
    print(w.fill(welcome_msg))
    print("\t" + "=" * 69)
    print()
    print(w.fill(description1))
    print()
    print(w.fill(description2))
    print()
    print(w.fill(description3))
    print()
    print(w.fill(description4))
    print()

    print("\tDo you understand, and are you ready to proceed? (yes/no)")
    print()
    consent = str(input("\t"))
    print()

    if consent == "no":
        raise Exception("User cancelled poll creation wizard!")

    print("\tExcellent! Let's begin!")

    print()

    print("\t" + "-" * 69)
    print()

    print(w.fill("Question 1: Input a question to ask in the poll"))
    print()
    print(w.fill("Examples:"))
    print("\t - What is your favorite type of food?")
    print("\t - What day shall we meet?")
    print("\t - Do you believe that climate change is a serious problem?")
    print()
    print()
    question = input("\t")
    print()

    print("\t" + "-" * 69)
    print()
    print(
        w.fill(
            "Question 2: Enter the number of choices, you would like to have in the poll"
        )
    )
    print()
    while True:
        try:
            num_choices = int(input("\t"))
        except ValueError:
            print()
            print(
                w.fill("Number of choices, should be an integer.Kindly re-enter again.")
            )
            print()
            continue
        break

    print()
    print("\t" + "-" * 69)
    print()
    print(w.fill("Excellent! Let's  input each choice for the input"))
    print()
    choices = []
    for idx in range(num_choices):
        print(w.fill(f"Enter Choice {idx+1}"))
        print()
        choice = str(input("\t"))
        choices.append(choice)
        print()
    print("\t" + "=" * 69)

    print()

    print(
        w.fill("All done! You have successfully completed the Poll Creation Wizard! 🎩")
    )
    return (question, choices)


def poll_answer_wizard(poll: ProjectMultipleChoicePoll) -> int:
    w = textwrap.TextWrapper(initial_indent="\t", subsequent_indent="\t")

    welcome_msg = "Welcome to the Poll Answer Wizard 🧙‍♂️ 🪄!!!"

    description1 = """You've arrived here because you were interested in a answering a poll.
A poll is a way to gather opinions or information from a group of people.
It typically involves asking a specific question and providing a set of answer choices for respondents to choose from"""

    description2 = """In a poll, the question is the inquiry being asked of the participants.
It should be a multiple-choice. The choices are the options that respondents
are given to select as their answer to the question.For example, a poll question might be
"Which is your favorite color?" with answer choices of red, blue, green, and yellow.
Participants can then select the answer that best represents their opinion or preference"""

    description3 = """Since you didn't pass in the choices into .answer_poll() (or you did so incorrectly),
this wizard is going to guide you through the process of answering the poll."""

    print("\t" + "=" * 69)
    print(w.fill(welcome_msg))
    print("\t" + "=" * 69)
    print()
    print(w.fill(description1))
    print()
    print(w.fill(description2))
    print()
    print(w.fill(description3))
    print()

    print("\tDo you understand, and are you ready to proceed? (yes/no)")
    print()
    consent = str(input("\t"))
    print()

    if consent == "no":
        raise Exception("User cancelled poll answer wizard!")

    print("\tExcellent! Let's display the poll question")

    print()

    print("\t" + "-" * 69)
    print()

    print(w.fill(f"Question : {poll.question}"))
    print()
    for idx, choice in enumerate(poll.choices):
        print(w.fill(f"{idx+1}. {choice}"))
        print()

    print("\t" + "-" * 69)
    print()

    print(w.fill("Kindly enter your choice for the poll"))
    print()
    while True:
        try:
            choice = int(input("\t"))
            if choice < 1 or choice > len(poll.choices):
                raise ValueError()
        except ValueError:
            print()
            print(
                w.fill(
                    f"Poll Answer should be a natural number between 1 and {len(poll.choices)}"
                )
            )
            print()
            continue
        break

    print("\t" + "=" * 69)
    print()
    print(w.fill("All done! You have successfully completed the Poll Answer Wizard! 🎩"))
    print()

    return choice


@serializable()
class AnswerProjectPoll(ProjectSubEvent):
    __canonical_name__ = "AnswerProjectPoll"
    __version__ = SYFT_OBJECT_VERSION_1

    answer: int


@serializable()
class ProjectMultipleChoicePoll(ProjectEventAddObject):
    __canonical_name__ = "ProjectPoll"
    __version__ = SYFT_OBJECT_VERSION_1

    question: str
    choices: List[str]
    allowed_sub_types: List[Type] = [AnswerProjectPoll]

    @validator("choices")
    def choices_min_length(cls, v):
        if len(v) < 1:
            raise ValueError("choices must have at least one item")
        return v

    def answer(self, answer: int) -> ProjectMessage:
        return AnswerProjectPoll(answer=answer, parent_event_id=self.id)

    def status(
        self, project: Project, pretty_print: bool = True
    ) -> Union[Dict, SyftError]:
        """Returns the status of the poll

        Args:
            project (Project): Project object to check the status

        Returns:
            str: Status of the poll

        During Poll calculation, a user would have answered the poll many times
        The status of the poll would be calculated based on the latest answer of the user
        """
        poll_answers = project.get_children(self)
        if len(poll_answers) == 0:
            return "No one has answered this poll"

        respondents = {}
        for poll_answer in poll_answers[::-1]:
            if not isinstance(poll_answer, AnswerProjectPoll):
                return SyftError(
                    message=f"Poll answer: {type(poll_answer)} is not of type AnswerProjectPoll"
                )
            creator_verify_key = poll_answer.creator_verify_key

            # Store only the latest response from the user
            identity = project.get_identity_from_key(creator_verify_key)
            if identity not in respondents:
                respondents[identity] = poll_answer.answer
        if pretty_print:
            for respondent, answer in respondents.items():
                print(f"{str(respondent.verify_key)[0:8]}: {answer}")
            print("\nChoices:\n")
            for idx, choice in enumerate(self.choices):
                print(f"{idx+1}: {choice}")
        else:
            return respondents


class ConsensusModel:
    pass


@serializable()
class DemocraticConsensusModel(ConsensusModel):
    threshold: float = 50

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, DemocraticConsensusModel):
            return False
        return self.threshold == value.threshold

    def __hash__(self) -> int:
        return hash(self.threshold)


def add_code_request_to_project(
    project: Union[ProjectSubmit, Project],
    code: SubmitUserCode,
    client: SyftClient,
    reason: Optional[str] = None,
) -> Union[SyftError, SyftSuccess]:
    if not isinstance(code, SubmitUserCode):
        return SyftError(
            message=f"Currently we are  only support creating requests for SubmitUserCode: {type(code)}"
        )

    if not isinstance(client, SyftClient):
        return SyftError(message="Client should be a valid SyftClient")

    if reason is None:
        reason = f"Code Request for Project: {project.name} has been submitted by {project.created_by}"

    submitted_req = client.api.services.code.request_code_execution(
        code=code, reason=reason
    )
    if isinstance(submitted_req, SyftError):
        return submitted_req

    request_event = ProjectRequest(linked_request=submitted_req)

    if isinstance(project, ProjectSubmit):
        project.bootstrap_events.append(request_event)
    else:
        result = project.add_event(request_event)
        if isinstance(result, SyftError):
            return result

    return SyftSuccess(
        message=f"Code request for '{code.func_name}' successfully added to '{project.name}' Project. "
        f"To see code requests by a client, run `[your_client].code`"
    )


@serializable()
class Project(SyftObject):
    __canonical_name__ = "Project"
    __version__ = SYFT_OBJECT_VERSION_1

    __repr_attrs__ = ["name", "description", "created_by"]
    __attr_unique__ = ["name"]

    # TODO: re-add users, members, leader_node_peer
    __hash_exclude_attrs__ = [
        "user_signing_key",
        "start_hash",
        "users",
        "members",
        "leader_node_peer",
        "event_id_hashmap",
    ]

    id: Optional[UID]
    name: str
    description: Optional[str]
    members: List[NodeIdentity]
    users: List[UserIdentity] = []
    username: Optional[str]
    created_by: str
    start_hash: Optional[str]
    # WARNING:  Do not add it to hash keys or print directly
    user_signing_key: Optional[SyftSigningKey] = None

    # Project events
    events: List[ProjectEvent] = []
    event_id_hashmap: Dict[UID, ProjectEvent] = {}

    # Project sync
    state_sync_leader: NodeIdentity
    leader_node_peer: Optional[NodePeer]

    # Unused
    consensus_model: ConsensusModel
    project_permissions: Set[str]
    # store: Dict[UID, Dict[UID, SyftObject]] = {}
    # permissions: Dict[UID, Dict[UID, Set[str]]] = {}

    def _coll_repr_(self):
        return {
            "name": self.name,
            "description": self.description,
            "created by": self.created_by,
            "pending requests": self.pending_requests,
        }

    def _repr_html_(self) -> Any:
        return (
            f"""
            <style>
            .syft-project {{color: {SURFACE[options.color_theme]};}}
            </style>
            """
            + "<div class='syft-project'>"
            + f"<h3>{self.name}</h3>"
            + f"<p>{self.description}</p>"
            + f"<p><strong>Created by: </strong>{self.username} ({self.created_by})</p>"
            + self.requests._repr_html_()
            + "<p>To see a list of projects, use command `&lt;your_client&gt;.projects`</p>"
            + "</div>"
        )

    def _broadcast_event(
        self, project_event: ProjectEvent
    ) -> Union[SyftSuccess, SyftError]:
        leader_client = self.get_leader_client(self.user_signing_key)

        return leader_client.api.services.project.broadcast_event(project_event)

    def get_all_identities(self) -> List[Identity]:
        return [*self.members, *self.users]

    def key_in_project(self, verify_key: SyftVerifyKey) -> bool:
        project_verify_keys = [
            identity.verify_key for identity in self.get_all_identities()
        ]

        return verify_key in project_verify_keys

    def get_identity_from_key(
        self, verify_key: SyftVerifyKey
    ) -> List[Union[NodeIdentity, UserIdentity]]:
        identities: List[Identity] = self.get_all_identities()
        for identity in identities:
            if identity.verify_key == verify_key:
                return identity
        return SyftError(message=f"Member with verify key: {verify_key} not found")

    def get_leader_client(self, signing_key: SyftSigningKey) -> SyftClient:
        if self.leader_node_peer is None:
            raise Exception("Leader node peer is not set")

        if signing_key is None:
            raise Exception("Signing key is required to create leader client")

        verify_key = signing_key.verify_key

        leader_client = SyftClientSessionCache.get_client_by_uid_and_verify_key(
            verify_key=verify_key, node_uid=self.leader_node_peer.id
        )

        if leader_client is None:
            leader_client = self.leader_node_peer.client_with_key(signing_key)
            SyftClientSessionCache.add_client_by_uid_and_verify_key(
                verify_key=verify_key,
                node_uid=leader_client.id,
                syft_client=leader_client,
            )

        return leader_client

    def has_permission(self, verify_key: SyftVerifyKey) -> bool:
        # Currently the permission function, initially checks only if the
        # verify key is present in the project
        # Later when the project object evolves, we can add more permission checks

        return self.key_in_project(verify_key)

    def _append_event(
        self, event: ProjectEvent, credentials: SyftSigningKey
    ) -> Union[SyftSuccess, SyftError]:
        prev_event = self.events[-1] if self.events else None
        valid = event.valid_descendant(self, prev_event)
        if not valid:
            return valid

        result = self._broadcast_event(event)
        if isinstance(result, SyftError):
            return result
        if isinstance(result, SyftNotReady):
            # If the client if out of sync, sync project updates from leader
            self.sync()
            event = event.rebase(project=self)
            event.sign(credentials)
            # Retrying broadcasting the event to leader
            # recursively call _append_event as due to network latency the event could reach late
            # and other events would be being streamed to the leader
            # This scenario could lead to starvation of node trying to sync with the leader
            # This would be solved in our future leaderless approach
            return self._append_event(event=event, credentials=credentials)

        event_copy = copy.deepcopy(event)
        self.events.append(event_copy)
        self.event_id_hashmap[event.id] = event_copy
        return result

    @property
    def event_ids(self) -> Iterable[UID]:
        return self.event_id_hashmap.keys()

    def add_event(
        self,
        event: ProjectEvent,
        credentials: Optional[Union[SyftSigningKey, SyftClient]] = None,
    ) -> Union[SyftSuccess, SyftError]:
        if event.id in self.event_ids:
            raise EventAlreadyAddedException(f"Event already added. {event}")

        if credentials is None:
            credentials = self.user_signing_key
        elif isinstance(credentials, SyftClient):
            credentials = credentials.credentials

        if not isinstance(credentials, SyftSigningKey):
            raise Exception(f"Adding an event requires a signing key. {credentials}")

        event.creator_verify_key = credentials.verify_key
        event._pre_add_update(self)
        event = event.rebase(self)
        event.sign(credentials)

        result = self._append_event(event, credentials=credentials)
        return result

    def validate_events(self, debug: bool = False) -> Union[SyftSuccess, SyftError]:
        current_hash = self.start_hash

        def valid_str(current_hash: int) -> str:
            return f"Project: {self.id} HEAD<{current_hash}> events are valid"

        if len(self.events) == 0:
            return SyftSuccess(message=valid_str(current_hash))

        last_event = None
        for event in self.events:
            result = event.valid_descendant(self, last_event)
            current_hash = event.event_hash

            if debug:
                icon = "✅" if result else "❌"
                prev_event = last_event if last_event is not None else self
                print(
                    f"{icon} {type(event).__name__}: {event.id} "
                    f"after {type(prev_event).__name__}: {prev_event.id}"
                )

            if not result:
                return result
            last_event = event
        return SyftSuccess(message=valid_str(current_hash))

    def get_children(self, event: ProjectEvent) -> List[ProjectEvent]:
        return self.get_events(parent_event_ids=event.id)

    def get_parent(self, parent_uid: UID) -> Optional[ProjectEvent]:
        parent_event = None
        event_query = self.get_events(ids=parent_uid)
        if len(event_query) == 0:
            return parent_event
        elif len(event_query) == 1:
            return event_query[0]
        else:
            raise Exception(f"More than 1 result for {parent_uid}")

    # TODO: add a another view for the project objects
    # to be able to have a Directed Acyclic Graph view of the graph events
    # this would allow to query the sub events effectively
    def get_events(
        self,
        types: Optional[Union[Type, List[Type]]] = None,
        parent_event_ids: Optional[Union[UID, List[UID]]] = None,
        ids: Optional[Union[UID, List[UID]]] = None,
    ):
        if types is None:
            types = []
        if isinstance(types, type):
            types = [types]

        if parent_event_ids is None:
            parent_event_ids = []
        if isinstance(parent_event_ids, UID):
            parent_event_ids = [parent_event_ids]

        if ids is None:
            ids = []
        if isinstance(ids, UID):
            ids = [ids]

        results = []
        for event in self.events:
            type_check = False
            if len(types) == 0 or isinstance(event, tuple(types)):
                type_check = True

            parent_check = False
            if (
                len(parent_event_ids) == 0 and not hasattr(event, "parent_event_id")
            ) or (
                hasattr(event, "parent_event_id")
                and event.parent_event_id in parent_event_ids
            ):
                parent_check = True

            id_check = False
            if len(ids) == 0 or event.id in ids:
                id_check = True

            if type_check and parent_check and id_check:
                results.append(event)
        return results

    def create_code_request(
        self,
        obj: SubmitUserCode,
        client: Optional[SyftClient] = None,
        reason: Optional[str] = None,
    ):
        if client is None:
            leader_client = self.get_leader_client(self.user_signing_key)
            res = add_code_request_to_project(
                project=self,
                code=obj,
                client=leader_client,
                reason=reason,
            )
            return res
        return add_code_request_to_project(
            project=self,
            code=obj,
            client=client,
            reason=reason,
        )

    def get_messages(self) -> List[Union[ProjectMessage, ProjectThreadMessage]]:
        messages = []
        for event in self.events:
            if isinstance(event, (ProjectMessage, ProjectThreadMessage)):
                messages.append(event)
        return messages

    @property
    def messages(self) -> str:
        message_text = ""
        top_messages = self.get_events(types=ProjectMessage)
        for message in top_messages:
            message_text += (
                f"{str(message.creator_verify_key)[0:8]}: {message.message}\n"
            )
            children = self.get_children(message)
            for child in children:
                message_text += (
                    f"> {str(child.creator_verify_key)[0:8]}: {child.message}\n"
                )
        if message_text == "":
            message_text = "No messages"
        return message_text

    def get_last_seq_no(self) -> int:
        return len(self.events)

    def send_message(self, message: str):
        message_event = ProjectMessage(message=message)
        result = self.add_event(message_event)
        if isinstance(result, SyftSuccess):
            return SyftSuccess(message="Message sent successfully")
        return result

    def reply_message(
        self,
        reply: str,
        message: Union[UID, ProjectMessage, ProjectThreadMessage],
    ):
        if isinstance(message, UID):
            if message not in self.event_ids:
                return SyftError(message=f"Message id: {message} not found")
            message = self.event_id_hashmap[message]

        reply_event: Union[ProjectMessage, ProjectThreadMessage]
        if isinstance(message, ProjectMessage):
            reply_event = message.reply(reply)
        elif isinstance(message, ProjectThreadMessage):
            reply_event = ProjectThreadMessage(
                message=reply, parent_event_id=message.parent_event_id
            )
        else:
            return SyftError(
                message=f"You can only reply to a message: {type(message)}"
                "Kindly re-check the msg"
            )

        result = self.add_event(reply_event)
        if isinstance(result, SyftSuccess):
            return SyftSuccess(message="Reply sent successfully")
        return result

    def create_poll(
        self,
        question: Optional[str] = None,
        choices: Optional[List[str]] = None,
    ):
        if (
            question is None
            or choices is None
            or not isinstance(question, str)
            or not isinstance(choices, list)
        ):
            question, choices = poll_creation_wizard()

        poll_event = ProjectMultipleChoicePoll(question=question, choices=choices)
        result = self.add_event(poll_event)
        if isinstance(result, SyftSuccess):
            return SyftSuccess(message="Poll created successfully")
        return result

    def answer_poll(
        self,
        poll: Union[UID, ProjectMultipleChoicePoll],
        answer: Optional[int] = None,
    ):
        if isinstance(poll, UID):
            if poll not in self.event_ids:
                return SyftError(message=f"Poll id: {poll} not found")
            poll = self.event_id_hashmap[poll]

        if not isinstance(poll, ProjectMultipleChoicePoll):
            return SyftError(
                message=f"You can only reply to a poll: {type(poll)}"
                "Kindly re-check the poll"
            )

        if not isinstance(answer, int) or answer <= 0 or answer > len(poll.choices):
            answer = poll_answer_wizard(poll)

        answer_event = poll.answer(answer)

        result = self.add_event(answer_event)
        if isinstance(result, SyftSuccess):
            return SyftSuccess(message="Poll answered successfully")
        return result

    def add_request(
        self,
        request: Request,
    ):
        linked_request = LinkedObject.from_obj(request, node_uid=request.node_uid)
        request_event = ProjectRequest(linked_request=linked_request)
        result = self.add_event(request_event)

        if isinstance(result, SyftSuccess):
            return SyftSuccess(message="Request created successfully")
        return result

    # Since currently we do not have the notion of denying a request
    # Adding only approve request, which would later be used to approve or deny a request
    def approve_request(
        self,
        request: Union[UID, ProjectRequest],
    ):
        if isinstance(request, UID):
            if request not in self.event_ids:
                return SyftError(message=f"Request id: {request} not found")
            request = self.event_id_hashmap[request]

        request_event: ProjectRequestResponse
        if isinstance(request, ProjectRequest):
            request_event = request.approve()
            if isinstance(request_event, SyftError):
                return request_event
        else:
            return SyftError(
                message=f"You can only approve a request: {type(request)}"
                "Kindly re-check the request"
            )
        result = self.add_event(request_event)
        if isinstance(result, SyftSuccess):
            return SyftSuccess(message="Request approved successfully")
        return result

    def sync(self, verbose: Optional[bool] = True) -> Union[SyftSuccess, SyftError]:
        """Sync the latest project with the state sync leader"""

        leader_client = self.get_leader_client(self.user_signing_key)

        unsynced_events = leader_client.api.services.project.sync(
            project_id=self.id, seq_no=self.get_last_seq_no()
        )
        if isinstance(unsynced_events, SyftError):
            return unsynced_events

        # UI progress bar for syncing
        if verbose and unsynced_events:
            with Progress() as progress:
                curr_val = 0
                task1 = progress.add_task(
                    f"[bold white]Syncing... {curr_val}/{len(unsynced_events)}",
                    total=len(unsynced_events),
                )

                while not progress.finished:
                    event = unsynced_events[curr_val]
                    curr_val += 1
                    progress.tasks[
                        task1
                    ].description = (
                        f"[bold white]Syncing... {curr_val}/{len(unsynced_events)}"
                    )
                    progress.update(task1, advance=1)
                    self.events.append(event)
                    self.event_id_hashmap[event.id] = event
                    # for a fancy UI view , deliberately slowing the sync
                    if curr_val <= 7:
                        time.sleep(0.1)
        else:
            for event in unsynced_events:
                self.events.append(event)
                self.event_id_hashmap[event.id] = event

        return SyftSuccess(message="Synced project  with Leader")

    @property
    def requests(self) -> List[Request]:
        return [
            event.request for event in self.events if isinstance(event, ProjectRequest)
        ]

    @property
    def pending_requests(self) -> int:
        return sum(
            [request.status == RequestStatus.PENDING for request in self.requests]
        )


@serializable(without=["bootstrap_events", "clients"])
class ProjectSubmit(SyftObject):
    __canonical_name__ = "ProjectSubmit"
    __version__ = SYFT_OBJECT_VERSION_1

    __hash_exclude_attrs__ = [
        "start_hash",
        "users",
        "members",
        "clients",
        "leader_node_route",
        "bootstrap_events",
    ]

    # stash rules
    __repr_attrs__ = ["name", "description", "created_by"]
    __attr_unique__ = ["name"]

    id: UID

    # Init args
    name: str
    description: Optional[str]
    members: Union[List[SyftClient], List[NodeIdentity]]

    # These will be automatically populated
    users: List[UserIdentity] = []
    created_by: Optional[str] = None
    username: Optional[str]
    clients: List[SyftClient] = []  # List of member clients
    start_hash: str = ""

    # Project sync args
    leader_node_route: Optional[NodeRoute]
    state_sync_leader: Optional[NodeIdentity]
    bootstrap_events: Optional[List[ProjectEvent]] = []

    # Unused at the moment
    project_permissions: Set[str] = set()
    consensus_model: ConsensusModel = DemocraticConsensusModel()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Preserve member SyftClients in a private variable clients
        # self.members will be List[NodeIdentity] on the node i.e. self.clients = []
        self.clients = self.get_syft_clients(self.members)

        # If ProjectSubmit is being re-created at node side
        if len(self.clients) == 0:
            return

        # Populate the created_by
        self.created_by = self.clients[0].logged_in_user

        # Extract information of logged in user from syft clients
        self.users = [UserIdentity.from_client(client) for client in self.clients]

        # Assign logged in user name as project creator
        self.username = self.clients[0].me.name or ""

        # Convert SyftClients to NodeIdentities
        self.members = list(map(self.to_node_identity, self.members))

    def _repr_html_(self) -> Any:
        return (
            f"""
            <style>
            .syft-project-create {{color: {SURFACE[options.color_theme]};}}
            </style>
            """
            + "<div class='syft-project-create'>"
            + f"<h3>{self.name}</h3>"
            + f"<p>{self.description}</p>"
            + f"<p><strong>Created by: </strong>{self.username} ({self.created_by})</p>"
            + "</div>"
        )

    @validator("members", pre=True)
    def verify_members(cls, val: Union[List[SyftClient], List[NodeIdentity]]):
        # SyftClients must be logged in by the same emails
        clients = cls.get_syft_clients(val)
        if len(clients) > 0:
            emails = {client.logged_in_user for client in clients}
            if len(emails) > 1:
                raise SyftException(
                    f"All clients must be logged in from the same account. Found multiple: {emails}"
                )
        return val

    @staticmethod
    def get_syft_clients(vals: Union[List[SyftClient], List[NodeIdentity]]):
        return [client for client in vals if isinstance(client, SyftClient)]

    @staticmethod
    def to_node_identity(val: Union[SyftClient, NodeIdentity]):
        if isinstance(val, NodeIdentity):
            return val
        elif isinstance(val, SyftClient):
            metadata = val.metadata.to(NodeMetadata)
            return metadata.to(NodeIdentity)
        else:
            raise SyftException(
                f"members must be SyftClient or NodeIdentity. Received: {type(val)}"
            )

    def create_code_request(
        self, obj: SubmitUserCode, client: SyftClient, reason: Optional[str] = None
    ):
        return add_code_request_to_project(
            project=self,
            code=obj,
            client=client,
            reason=reason,
        )

    def start(self, return_all_projects=False) -> Project:
        # Currently we are assuming that the first member is the leader
        # This would be changed in our future leaderless approach
        leader = self.clients[0]
        followers = self.clients[1:]

        try:
            # Check for DS role across all members
            self._pre_submit_checks(self.clients)

            # Exchange route between leaders and followers
            self._exchange_routes(leader, followers)

            # create project for each node
            projects_map = self._create_projects(self.clients)

            # bootstrap project with pending events on leader node's project
            self._bootstrap_events(projects_map[leader])

            if return_all_projects:
                return list(projects_map.values())

            return projects_map[leader]
        except SyftException as exp:
            return SyftError(message=str(exp))

    def _pre_submit_checks(self, clients: List[SyftClient]):
        try:
            # Check if the user can create projects
            for client in clients:
                result = client.api.services.project.can_create_project()
                if isinstance(result, SyftError):
                    raise SyftException(result.message)
        except Exception:
            raise SyftException("Only Data Scientists can create projects")

        return True

    def _exchange_routes(self, leader: SyftClient, followers: List[SyftClient]):
        # Since we are implementing a leader based system
        # To be able to optimize exchanging routes.
        # We require only the leader to exchange routes with all the members
        # Meaning if we could guarantee, that the leader node is able to reach the members
        # the project events could be broadcasted to all the members

        for follower in followers:
            result = leader.exchange_route(follower)
            if isinstance(result, SyftError):
                raise SyftException(result.message)

        self.leader_node_route = connection_to_route(leader.connection)

    def _create_projects(self, clients: List[SyftClient]):
        projects: Dict[SyftClient, Project] = {}

        for client in clients:
            result = client.api.services.project.create_project(project=self)
            if isinstance(result, SyftError):
                raise SyftException(result.message)
            projects[client] = result

        return projects

    def _bootstrap_events(self, leader_project: Project):
        if not self.bootstrap_events:
            return

        while len(self.bootstrap_events) > 0:
            event = self.bootstrap_events.pop(0)
            result = leader_project.add_event(event)
            if isinstance(result, SyftError):
                raise SyftException(result.message)


def add_members_as_owners(members: List[SyftVerifyKey]) -> Set[str]:
    keys = set()
    for member in members:
        owner_key = f"OWNER_{member.verify_key}"
        keys.add(owner_key)
    return keys


def elect_leader(context: TransformContext) -> TransformContext:
    if len(context.output["members"]) == 0:
        raise Exception("Project's require at least one member")

    context.output["state_sync_leader"] = context.output["members"][0]

    return context


def check_permissions(context: TransformContext) -> TransformContext:
    if len(context.output["members"]) > 1:
        # more than 1 node
        pass

    # check at least one owner
    if len(context.output["project_permissions"]) == 0:
        project_permissions = context.output["project_permissions"]
        project_permissions = project_permissions.union(
            add_members_as_owners(context.output["members"])
        )
        context.output["project_permissions"] = project_permissions

    return context


def add_creator_name(context: TransformContext) -> TransformContext:
    context.output["username"] = context.obj.username
    return context


@transform(ProjectSubmit, Project)
def new_projectsubmit_to_project() -> List[Callable]:
    return [elect_leader, check_permissions, add_creator_name]


def hash_object(obj: Any) -> Tuple[bytes, str]:
    """Hashes an object using sha256

    Args:
        obj (Any): Object to be hashed

    Returns:
        str: Hashed value of the object
    """
    hash_bytes = _serialize(obj, to_bytes=True, for_hashing=True)
    hash = hashlib.sha256(hash_bytes)
    return (hash.digest(), hash.hexdigest())


def create_project_hash(project: Project) -> Tuple[bytes, str]:
    # Creating a custom hash for the project
    # as the recursive hash is yet to be revamped
    # for primitives python types

    # hashing is calculated based on the following attributes
    # attrs = ["name", "description", "created_by", "members", "users"]

    return hash_object(
        [
            project.name,
            project.description,
            project.created_by,
            [hash_object(member) for member in project.members],
            [hash_object(user) for user in project.users],
        ]
    )


def create_project_event_hash(project_event: ProjectEvent) -> Tuple[bytes, str]:
    # Creating a custom hash for the project
    # as the recursive hash is yet to be revamped
    # for primitives python types.

    # hashing is calculated based on the following attributes,
    # attrs = ["id", "project_id", "seq no",
    # "prev_event_uid", "prev_event_hash", "creator_verify_key"]

    return hash_object(
        [
            project_event.id,
            project_event.project_id,
            project_event.seq_no,
            project_event.prev_event_uid,
            project_event.timestamp.utc_timestamp,
            project_event.prev_event_hash,
            hash_object(project_event.creator_verify_key)[1],
        ]
    )
