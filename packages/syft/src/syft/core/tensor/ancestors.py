# future
from __future__ import annotations

# stdlib
import textwrap
from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

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

_PhiTensorRef = None


def _PhiTensor() -> Type[PassthroughTensor]:
    global _PhiTensorRef
    if _PhiTensorRef is None:
        # relative
        from .autodp.phi_tensor import PhiTensor

        _PhiTensorRef = PhiTensor
    return _PhiTensorRef


def data_subject_creation_wizard(data: Any) -> List[Any]:

    w = textwrap.TextWrapper(initial_indent="\t", subsequent_indent="\t")

    welcome_msg = "Welcome to the Data Subject Annotation Wizard!!!"

    description1 = """You've arrived here because you called Tensor.private() without passing in any data_subjects!
Since the purpose of .private() is to add metadata for the support of automatic differential
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

    description3 = """Since you didn't pass in data_subjects into .private() (or you did so incorrectly), this wizard is
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
        print(w.fill("\t.private(data_subjects='" + str(single_uid) + "')"))
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
                    "All done! Next time if you want to skip the wizard, call .private() like this:"
                )
            )
            print()
            print(
                w.fill(
                    ".private(data_subjects=['"
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
                    "Excellent. Well, in that case you'll need to re-run .private() but pass in"
                    " a list of strings where each string is a unique identifier for an data subject, and where"
                    " the length of the list is equal to the number of rows in your tensor. Like so:"
                )
            )

            print()
            print(w.fill(".private(data_subjects=['bob', 'alice', 'john'])"))
            print()
            print(
                " Now just to make sure I don't corrupt your tensor - I'm going to throw an exception."
            )
            print()
            raise Exception(
                "Wizard aborted. Please run .private(data_subjects=<your data_subjects>)"
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
                "\t\ttensor.private(min_vals=0, max_vals=1, data_subjects=data_subjects))"
            )
            print()
            print(
                "Aborting wizard now so that you rcan re-run .private with the right parameters."
            )
            print()
            raise Exception(
                "Wizard aborted. Please run .private(data_subjects=<your data_subjects>)"
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
    def min_vals(self):  # type: ignore
        return self.__class__(self.child.min_vals)

    @property
    def max_vals(self):  # type: ignore
        return self.__class__(self.child.max_vals)

    @property
    def gamma(self):  # type: ignore
        return self.__class__(self.child.gamma)

    def publish(
        self,
        get_budget_for_user: Callable,
        deduct_epsilon_for_user: Callable,
        ledger: DataSubjectLedger,
        sigma: float,
    ) -> Any:
        return self.child.publish(
            get_budget_for_user, deduct_epsilon_for_user, ledger, sigma
        )

    def copy(self) -> PhiTensorAncestor:
        """This should certainly be implemented by the subclass but adding this here to
        satisfy mypy."""

        return NotImplemented

    def annotated_with_dp_metadata(
        self,
        min_val: ArrayLike,
        max_val: ArrayLike,
        data_subjects: Optional[Any] = None,
        skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:
        print("Tensor annotated with DP Metadata")
        return self.private(
            min_val=min_val,
            max_val=max_val,
            data_subjects=data_subjects,
            skip_blocking_checks=skip_blocking_checks,
        )

    def private(
        self,
        min_val: ArrayLike,
        max_val: ArrayLike,
        data_subjects: Optional[Any] = None,
        skip_blocking_checks: bool = False,
    ) -> PhiTensorAncestor:
        return self.copy()._private(
            min_val=min_val,
            max_val=max_val,
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
                "At present, you can only call .private() "
                + "on syft.Tensor objects wrapping numpy arrays. You called it on a "
                + "syft.Tensor wrapping a "
                + str(type(self.child))
            )

            if isinstance(self.child, np.ndarray):
                msg += " with dtype:" + str(getattr(self.child, "dtype", None))

            raise TypeError(msg)

        # Check 2: If data_subjects == None, then run the entity creation tutorial
        if data_subjects is None:
            if skip_blocking_checks:
                raise Exception(
                    "Error: 'data_subjects' argument to .private() must not be None!"
                )
            print(
                "ALERT: You didn't pass in any data_subjects. Launching data subject wizard...\n"
            )
            data_subjects = data_subject_creation_wizard(self.child)

        # Check 3: If data_subjects is a string, make it a list with one entity in it
        if isinstance(data_subjects, str):
            data_subjects = [DataSubjectArray(data_subjects)]
        elif isinstance(data_subjects, DataSubjectArray):
            data_subjects = [data_subjects]
        # Check 4: If data_subjects are a list, are the items strings or DataSubjectArray objects.
        # If they're strings lets create DataSubjectArray objects.

        # if isinstance(data_subjects, (list, tuple)):
        #     data_subjects = np.array(data_subjects)

        # if len(data_subjects) != 1 and data_subjects.shape != self.shape:
        #     raise Exception(
        #         "Entities shape doesn't match data shape. If you're"
        #         " going to pass in something other than 1 entity for the"
        #         " entire tensor or one entity per row, you're going to need"
        #         " to make the np.ndarray of data_subjects have the same shape as"
        #         " the tensor you're calling .private() on. Try again."
        #     )

        if not isinstance(data_subjects, DataSubjectArray):
            data_subjects = DataSubjectArray.from_objs(data_subjects)

        # SKIP check temporarily
        # for entity in one_hot_lookup:
        #     if not isinstance(entity, (np.integer, str, DataSubject)):
        #         raise ValueError(
        #             f"Expected DataSubject to be either string or DataSubject object, but type is {type(entity)}"
        #         )
        if data_subjects.shape != self.shape:
            raise ValueError(
                f"DataSubjects shape: {data_subjects.shape} should match data shape: {self.shape}"
            )

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

        if min_vals.shape != self.child.shape:
            min_vals = lazyrepeatarray(min_vals, self.child.shape)

        if max_vals.shape != self.child.shape:
            max_vals = lazyrepeatarray(max_vals, self.child.shape)

        self.replace_abstraction_top(
            tensor_type=_PhiTensor(),
            child=self.child,
            min_vals=min_vals,
            max_vals=max_vals,
            data_subjects=data_subjects,  # type: ignore
        )  # type: ignore

        return self
