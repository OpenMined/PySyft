# future
from __future__ import annotations

# stdlib
import textwrap
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Tuple
from typing import Type
from warnings import warn

# third party
import numpy as np
from numpy.typing import ArrayLike

# relative
from ..adp.data_subject_ledger import DataSubjectLedger
from ..adp.data_subject_list import DataSubjectArray
from .config import DEFAULT_FLOAT_NUMPY_TYPE
from .config import DEFAULT_INT_NUMPY_TYPE
from .lazy_repeat_array import lazyrepeatarray
from .manager import TensorChainManager
from .passthrough import PassthroughTensor  # type: ignore

if TYPE_CHECKING:
    # relative
    from .autodp.gamma_tensor import GammaTensor

_PhiTensorRef = None
_GammaTensorRef = None


def _PhiTensor() -> Type[PassthroughTensor]:
    global _PhiTensorRef
    if _PhiTensorRef is None:
        # relative
        from .autodp.phi_tensor import PhiTensor

        _PhiTensorRef = PhiTensor
    return _PhiTensorRef


def _GammaTensor() -> Type[GammaTensor]:
    global _GammaTensorRef
    if _GammaTensorRef is None:
        # relative
        from .autodp.gamma_tensor import GammaTensor

        _GammaTensorRef = GammaTensor
    return _GammaTensorRef


def data_subject_creation_wizard(data: Any) -> List[Any]:

    w = textwrap.TextWrapper(initial_indent="\t", subsequent_indent="\t")

    welcome_msg = "Welcome to the Data Subject Annotation Wizard!!!"

    description1 = """You've arrived here because you called Tensor.annotate_with_dp_metadata() without passing in any data_subjects!
Since the purpose of .annotate_with_dp_metadata() is to add metadata for the support of automatic differential
privacy budgeting, you need to describe which parts of your Tensor correspond to which
real-world data subjects (data_subjects) whose privacy you want to protect. This is the only
way the system knows, for example, that it costs twice as much privacy budget when twice
as much of your data (say, 2 rows instead of 1 row) refer to the same data subject."""

    description2 = """Entities can be people (such as a medical patient), places (such as a family's address), or
even organizations (such as a business, state, or country). If you're not sure what kind of data subject
to include, just ask yourself the question, "who am I trying to protect the privacy of?". If it's
an organization, make one data subject per organization. If it's people, make one data subject per person.
If it's a group of people who are somehow similar/linked to each other (such as a family),
make each data subject a different group. For more information on differential privacy, see OpenMined's
course on the subject: https://courses.openmined.org/"""

    description3 = """Since you didn't pass in data_subjects into .annotate_with_dp_metadata() (or you did so incorrectly), this wizard is
going to guide you through the process of annotating your data with data_subjects."""

    description4 = """In this wizard, we're going to ask you for *unique identifiers* which refer to the data_subjects
in your data. While the unique identifiers need not be personal data (they can be random strings of letters and numbers
if you like). It is ESSENTIAL that you use the same identifier when referring to the same data subject in the
data that you never accidentally refer to two data_subjects by the same identifier. Additionally, if you plan
to do any kind of data JOIN with another dataset, it is ESSENTIAL that you are using the same unique
identifiers for data_subjects as the data you're joining with. Since these unique identifiers may be personal
information, PySyft might not be able to detect if two tensors are using different identifiers for the
same person."""

    description5 = """So, in this tutorial we're going to be asking you to specify Unique Identifiers (UIDs) for each
data subject in your data. This could be an email, street address, or any other string that identifies someone
uniquely in your data and in the data you intend to use with your data (if any)."""

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
    print(w.fill(description5))
    print()

    print("\tDo you understand, and are you ready to proceed? (yes/no)")
    print()
    consent = str(input("\t"))
    print()

    if consent == "no":
        raise Exception("User cancelled data subject creation wizard!")

    print("\tExcellent! Let's begin!")
    # print("\tYou passed in a tensor with the shape:" + str(data.shape))
    print()

    print("\t" + "-" * 69)
    print()

    print(
        w.fill("Question 1: Is this entire tensor referring to the same data subject?")
    )
    print()
    print(w.fill("Examples:"))
    print("\t - a single medical scan of one patient")
    print("\t - a single spreadsheet of proprietary statistics about a business")
    print("\t - a tensor of facts about a country")
    print()
    print(
        w.fill(
            """(if the tensor is about one data subject, but it also contains multiple other data_subjects within,
such as a tensor about all the customers of one business, ask yourself, are you trying to
protect the people or the business)"""
        )
    )
    print()
    print(
        w.fill(
            "If yes, write the UID of the data subject this data is about, otherwise write 'no' "
            " because this data is about more than one data subject."
        )
    )
    print()
    single_uid = input("\t")
    print()
    if single_uid != "no":
        print("\t" + "-" * 69)
        print()
        print(
            w.fill(
                "Excellent! Your data will be annotated as referring to:"
                + str(single_uid)
            )
        )
        print()
        print(
            w.fill(
                "Congratulations! You're all done with the Data Subject Annotation Wizard!!!"
                "In the future, you can accomplish this without the wizard by running:"
            )
        )
        print()
        print(
            w.fill(
                "\t.annotate_with_dp_metadata(data_subjects='" + str(single_uid) + "')"
            )
        )
        print()
        print("\t" + "=" * 69)
        return [single_uid]

    print("\t" + "-" * 69)
    print()
    print(
        w.fill(
            "Question 2: Does each row correspond to an data subject, perhaps with occasional repeats (yes/no)?"
        )
    )
    print()
    answer = str(input("\t"))
    print()
    print("\t" + "-" * 69)
    print()
    if answer == "yes":
        print(
            w.fill(
                "Question 3: Excellent! Well, since your dataset has "
                + str(data.shape[0])
                + " rows, "
                + "would you like to hand enter an data subject for each one (yes) or if there are too "
                + "many for you to hand-enter, we'll print some example code for you to run (no)."
            )
        )

        print()

        answer = str(input("\t"))

        if answer == "yes":

            print()

            data_subjects = list()
            for i in range(len(data)):
                print("\t\t" + "-" * 61)
                print()
                print(w.fill("\tData Row " + str(i) + ":" + str(data[i])))
                ent = input("\t\t What data subject is this row about:")
                data_subjects.append(ent)
                print()
            print("\t\t" + "-" * 61)
            print()
            print(
                w.fill(
                    "All done! Next time if you want to skip the wizard, call .annotate_with_dp_metadata() like this:"
                )
            )
            print()
            print(
                w.fill(
                    ".annotate_with_dp_metadata(data_subjects=['"
                    + data_subjects[0]
                    + "', '"
                    + data_subjects[1]
                    + "', '"
                    + data_subjects[-1]
                    + "'])"
                )
            )
            print()
            print(
                w.fill(
                    " where you pass in data_subjects as a list of strings, one per row. As long as you"
                    " pass in the same number of data_subjects as there are rows in your tensor, it will"
                    " automatically detect you have and assume you mean one data subject per row."
                )
            )
            return data_subjects

        elif answer == "no":

            print()

            print(
                w.fill(
                    "Excellent. Well, in that case you'll need to re-run .annotate_with_dp_metadata() but pass in"
                    " a list of strings where each string is a unique identifier for an data subject, and where"
                    " the length of the list is equal to the number of rows in your tensor. Like so:"
                )
            )

            print()
            print(
                w.fill(
                    ".annotate_with_dp_metadata(data_subjects=['bob', 'alice', 'john'])"
                )
            )
            print()
            print(
                " Now just to make sure I don't corrupt your tensor - I'm going to throw an exception."
            )
            print()
            raise Exception(
                "Wizard aborted. Please run .annotate_with_dp_metadata(data_subjects=<your data_subjects>)"
                " again with your list of data subject unique identifiers (strings),"
                "one per row of your tensor."
            )
    elif answer == "no":

        print(
            w.fill(
                "Question 3: Is your data one data subject for every column (yes/no)?"
            )
        )

        print()

        answer = str(input("\t"))

        print()

        if answer == "yes":
            print(
                w.fill(
                    "We don't yet support this form of injestion. Please transpose your data"
                    " into one data subject per row and re-run the wizard. Aborting:)"
                )
            )

            raise Exception("Wizard aborted.")

        elif answer == "no":

            print(
                w.fill(
                    "It sounds like your tensor is a random assortment of data_subjects (and perhaps empty/non"
                    "-data_subjects). If you have empty values, just create random data_subjects for them for now. If "
                    "you have various data_subjects scattered throughout your tensor (not organized by row), then "
                    "you'll need to pass in a np.ndarray of strings which is identically shaped to your data in data "
                    "subjects like so:"
                )
            )

            print()
            print("\t\ttensor = sy.Tensor(np.ones((2,2)).astype(np.int32))")
            print()
            print(
                "\t\tdata_subjects = np.array([['bob', 'alice'],['charlie', 'danielle']])"
            )
            print()
            print(
                "\t\ttensor.annotate_with_dp_metadata(min_vals=0, max_vals=1, data_subjects=data_subjects))"
            )
            print()
            print(
                "Aborting wizard now so that you rcan re-run .private with the right parameters."
            )
            print()
            raise Exception(
                "Wizard aborted. Please run .annotate_with_dp_metadata(data_subjects=<your data_subjects>)"
                " again with your np.ndarray of data subject unique identifiers (strings),"
                " one per value of your tensor and where your np.ndarray of data_subjects is"
                " the same shape as your data."
            )
    print()

    print("\t" + "_" * 69)
    raise Exception(
        "Not sure what happened... this code shouldn't have been reached. Try answering questions with "
        "options given by the prompts (such as yes/no)."
    )


class PhiTensorAncestor(TensorChainManager):
    """Inherited by any class which might have or like to have SingleEntityPhiTensor in its chain
    of .child objects"""

    def __init__(self, child: Any) -> None:
        self.child = child

    @property
    def shape(self) -> Tuple[Any, ...]:
        return self.child.shape

    @property
    def size(self) -> int:
        return self.child.size

    @property
    def min_vals(self):  # type: ignore
        return self.child.min_vals

    @property
    def max_vals(self):  # type: ignore
        return self.child.max_vals

    @property
    def gamma(self):  # type: ignore
        return self.__class__(self.child.gamma)

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
        private: bool,
    ) -> Any:
        return self.child.publish(
            get_budget_for_user, deduct_epsilon_for_user, ledger, sigma, private=private
        )

    def copy(self) -> PhiTensorAncestor:
        """This should certainly be implemented by the subclass but adding this here to
        satisfy mypy."""

        return NotImplemented

    def private(
        self,
        min_val: ArrayLike,
        max_val: ArrayLike,
        data_subjects: Optional[Any] = None,
        skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:
        """[DEPRECATED] This method will annotate your Tensor with metadata (an upper bound
        and lower bound on the data, as well as the people whose data is in the dataset),
        and thus enable Differential Privacy protection.

        Note: Deprecated in 0.7.0
        `.private` method will be removed in 0.8.0, it is replaced by `annotate_with_dp_metadata`.
        """

        _deprec_message = (
            "This method is deprecated in v0.7.0 and will be removed in future version updates. "
            "It is replaced with `annotate_with_dp_metadata` to provide a user-friendly experience. "
            "One can call `help(syft.Tensor.annotate_with_dp_metadata)` to learn more about its use."
        )

        warn(_deprec_message, DeprecationWarning, stacklevel=2)

        return self.annotate_with_dp_metadata(
            lower_bound=min_val,
            upper_bound=max_val,
            data_subjects=data_subjects,
            skip_blocking_checks=skip_blocking_checks,
        )

    def annotated_with_dp_metadata(
        self,
        min_val: ArrayLike,
        max_val: ArrayLike,
        data_subjects: Optional[Any] = None,
        skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:
        """[DEPRECATED] This method will annotate your Tensor with metadata (an upper bound
        and lower bound on the data, as well as the people whose data is in the dataset),
        and thus enable Differential Privacy protection.

        Note: Deprecated in 0.7.0
        `.annotated_with_dp_metadata` method will be removed in 0.8.0, it is renamed to `annotate_with_dp_metadata`.
        """

        _deprec_message = (
            "This method is deprecated in v0.7.0 and will be removed in future version. "
            "It is renamed to `annotate_with_dp_metadata` with function arguments `min_val` and `max_val` "
            "renamed to `lower_bound` and `upper_bound` respectively. This has been done to simplify the definition "
            "of the function in use. "
            "One can call `help(syft.Tensor.annotate_with_dp_metadata)` to learn more about its use."
        )

        warn(_deprec_message, DeprecationWarning, stacklevel=2)

        return self.annotate_with_dp_metadata(
            lower_bound=min_val,
            upper_bound=max_val,
            data_subjects=data_subjects,
            skip_blocking_checks=skip_blocking_checks,
        )

    def annotate_with_dp_metadata(
        self,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        data_subjects: Optional[Any] = None,
        skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:
        """
        This method will annotate your Tensor with metadata (an upper bound and lower bound on the data,
        as well as the people whose data is in the dataset), and thus enable Differential Privacy
        protection.

        Params:
            lower_bound: float
                The lowest possible value allowed by your dataset's schema.

                e.g.
                - if this is data about age, lower_bound would ideally be 0.
                - if this is data about teenagers' ages, lower_bound would ideally be 13.
                - if these are RGB images, lower_bound would ideally be 0.

            upper_bound: float
                The highest possible value allowed by your dataset's schema.

                e.g.
                - if this is data about age, upper_bound would ideally be 120.
                (the age of the oldest known human)
                - if this is data about teenagers' ages, upper_bound would ideally be 19.
                (the oldest possible teenager)
                - if these are RGB images, upper_bound would ideally be 255.


            data_subjects: str, tuple, list, np.ndarray, DataSubjectArray
                The individuals whose data is in this dataset, and whose privacy you wish to protect.

                Can be either:
                 - string: data_subjects="Bob"
                 - list: data_subjects=["Bob", "Alice", "Joe"]
                 - tuple: data_subjects=("Bob", "Alice")
                 - array: data_subjects=np.array(["Bob", "Alice", "Joe"])

                Please provide either:
                    - 1 data subject for the whole dataset
                        i.e. This is one person's finances. The dataset has shape=(10, 10), and we provide
                        data_subject="Bob"

                    - 1 data subjects per row
                        i.e. This dataset is 5 images of size (28, 28) and thus has a shape of (5, 28, 28), we provide
                        data_subjects=["Bob", "Alice", "Julian", "Billy", "Chris"]

                    - 1 data subject per each data point
                        i.e. This dataset has people's ages and has shape (1000,) and we provide
                        data_subjects=np.arange(1000)

        Returns:
            Syft Tensor
                This tensor can be protected by PySyft's differential Privacy System.

        If this documentation is not clear- please feel free to post in the #support channel on Slack.
        You may join here: https://slack.openmined.org/
        """

        print("Tensor annotated with DP Metadata!")
        print(
            "You can upload this Tensor to a domain node by calling `<domain_client>.load_dataset` "
            "and passing in this tensor as an asset."
        )

        return self.copy()._private(
            min_val=lower_bound,
            max_val=upper_bound,
            data_subjects=data_subjects,
            skip_blocking_checks=skip_blocking_checks,
        )

    def _private(
        self,
        min_val: ArrayLike,
        max_val: ArrayLike,
        data_subjects: Optional[Any] = None,
        skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:
        # PHASE 1: RUN CHECKS

        # Check 1: Is self.child a compatible type? We only support DP and SMPC for a few types.
        if not isinstance(self.child, np.ndarray):

            msg = (
                "At present, you can only call .annotate_with_dp_metadata() "
                + "on syft.Tensor objects wrapping numpy arrays. You called it on a "
                + "syft.Tensor wrapping a "
                + str(type(self.child))
            )

            if isinstance(self.child, np.ndarray):
                msg += " with dtype:" + str(getattr(self.child, "dtype", None))

            raise TypeError(msg)

        # Check 2: Have Data Subjects been allotted properly?
        data_subjects = check_data_subjects(
            data=self.child,
            data_subjects=data_subjects,
            skip_blocking_checks=skip_blocking_checks,
        )

        # Check 3: Min/Max val metadata
        min_vals, max_vals = check_min_max_vals(
            min_val, max_val, target_shape=self.child.shape
        )

        if any(len(x.item()) > 1 for x in np.nditer(data_subjects, flags=["refs_ok"])):
            self.replace_abstraction_top(
                tensor_type=_GammaTensor(),
                child=self.child,
                min_vals=min_vals,
                max_vals=max_vals,
                data_subjects=data_subjects,  # type: ignore
            )  # type: ignore
        else:
            self.replace_abstraction_top(
                tensor_type=_PhiTensor(),
                child=self.child,
                min_vals=min_vals,
                max_vals=max_vals,
                data_subjects=data_subjects,  # type: ignore
            )  # type: ignore

        return self


def check_data_subjects(
    data: np.ndarray, data_subjects: Optional[Any], skip_blocking_checks: bool
) -> np.ndarray:
    # Check 2: If data_subjects == None, then run the entity creation tutorial
    if data_subjects is None:
        if skip_blocking_checks:
            raise Exception(
                "Error: 'data_subjects' argument to .annotate_with_dp_metadata() must not be None!"
            )
        print(
            "ALERT: You didn't pass in any data_subjects. Launching data subject wizard...\n"
        )
        data_subjects = data_subject_creation_wizard(data)

    # Check 3: If data_subjects is a string, make it an array of the correct shape
    if isinstance(data_subjects, str):
        data_subjects = np.array(
            DataSubjectArray.from_objs([data_subjects] * data.size)
        )
        data_subjects = data_subjects.reshape(data.shape)

    if isinstance(data_subjects, DataSubjectArray):
        # if data.size == 1, data_subjects will be a DSA instead of a np array
        data_subjects = np.array(data_subjects)

    if not isinstance(data_subjects, DataSubjectArray):
        data_subjects = DataSubjectArray.from_objs(data_subjects)

    if data_subjects.shape != data.shape:
        if data_subjects.size == 1:
            # 1 data subject for the entire tensor.
            data_subjects = np.broadcast_to(data_subjects, data.shape)
        elif data_subjects.shape[0] == data.shape[0] and len(data_subjects.shape) == 1:
            # e.g. 1 data subject per image for 10 imgs: child = (10, 25, 25) and data_subjects = (10,)

            axis_count = len(data.shape)
            axes_to_expand = list(range(axis_count))[1:]
            data_subjects = np.expand_dims(data_subjects, axis=axes_to_expand)

            data_subjects = np.broadcast_to(data_subjects, data.shape)
        else:
            raise Exception(
                "Data Subject shape doesn't match the shape of your data. Please provide either 1 data subject per "
                "data point, or 1 data subject per row of data. Please try again."
            )
    return data_subjects


def check_min_max_vals(
    min_val: ArrayLike, max_val: ArrayLike, target_shape: Tuple
) -> Tuple[lazyrepeatarray, lazyrepeatarray]:
    if isinstance(min_val, (bool, int, float)):
        min_vals = np.array(min_val).ravel()  # make it 1D
        if isinstance(min_val, int):
            min_vals = min_vals.astype(DEFAULT_INT_NUMPY_TYPE)  # type: ignore
        if isinstance(min_val, float):
            min_vals = min_vals.astype(DEFAULT_FLOAT_NUMPY_TYPE)  # type: ignore
    else:
        raise Exception(
            "min_vals should be either float,int,bool got "
            + str(type(min_val))
            + " instead."
        )

    if isinstance(max_val, (bool, int, float)):
        max_vals = np.array(max_val).ravel()  # make it 1D
        if isinstance(max_val, int):
            max_vals = max_vals.astype(DEFAULT_INT_NUMPY_TYPE)  # type: ignore
        if isinstance(max_val, float):
            max_vals = max_vals.astype(DEFAULT_FLOAT_NUMPY_TYPE)  # type: ignore
    else:
        raise Exception(
            "min_vals should be either float,int,bool got "
            + str(type(max_val))
            + " instead."
        )

    if min_vals.shape != target_shape:
        min_vals = lazyrepeatarray(min_vals, target_shape)

    if max_vals.shape != target_shape:
        max_vals = lazyrepeatarray(max_vals, target_shape)
    return min_vals, max_vals
