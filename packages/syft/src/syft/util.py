# stdlib
import os
from pathlib import Path
from secrets import randbelow
from typing import Any
from typing import List
from typing import Optional
from typing import Union

# third party
from forbiddenfruit import curse
from nacl.signing import SigningKey
from nacl.signing import VerifyKey

# syft absolute
import syft

# relative
from .logger import critical
from .logger import debug
from .logger import error
from .logger import traceback_and_raise


def validate_type(_object: object, _type: type, optional: bool = False) -> Any:
    if isinstance(_object, _type) or (optional and (_object is None)):
        return _object

    traceback_and_raise(
        f"Object {_object} should've been of type {_type}, not {_object}."
    )


def validate_field(_object: object, _field: str) -> Any:
    object = getattr(_object, _field, None)

    if object is not None:
        return object

    traceback_and_raise(f"Object {_object} has no {_field} field set.")


def get_subclasses(obj_type: type) -> List[type]:
    """Recursively generate the list of all classes within the sub-tree of an object

    As a paradigm in Syft, we often allow for something to be known about by another
    part of the codebase merely because it has subclassed a particular object. While
    this can be a big "magicish" it also can simplify future extensions and reduce
    the likelihood of small mistakes (if done right).

    This is a utility function which allows us to look for sub-classes and the sub-classes
    of those sub-classes etc. returning a full list of descendants of a class in the inheritance
    hierarchy.

    Args:
        obj_type: the type we want to look for sub-classes of

    Returns:
        the list of subclasses of obj_type

    """

    classes = list()
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes


def index_modules(a_dict: object, keys: List[str]) -> object:
    """Recursively find a syft module from its path

    This is the recursive inner function of index_syft_by_module_name.
    See that method for a full description.

    Args:
        a_dict: a module we're traversing
        keys: the list of string attributes we're using to traverse the module

    Returns:
        a reference to the final object

    """

    if len(keys) == 0:
        return a_dict
    return index_modules(a_dict=a_dict.__dict__[keys[0]], keys=keys[1:])


def index_syft_by_module_name(fully_qualified_name: str) -> object:
    """Look up a Syft class/module/function from full path and name

    Sometimes we want to use the fully qualified name (such as one
    generated from the 'get_fully_qualified_name' method below) to
    fetch an actual reference. This is most commonly used in deserialization
    so that we can have generic protobuf objects which just have a string
    representation of the specific object it is meant to deserialize to.

    Args:
        fully_qualified_name: the name in str of a module, class, or function

    Returns:
        a reference to the actual object at that string path

    """

    # @Tudor this needs fixing during the serde refactor
    # we should probably just support the native type names as lookups for serde
    if fully_qualified_name == "builtins.NoneType":
        fully_qualified_name = "syft.lib.python._SyNone"
    attr_list = fully_qualified_name.split(".")

    # we deal with VerifyAll differently, because we don't it be imported and used by users
    if attr_list[-1] == "VerifyAll":
        return type(syft.core.common.group.VERIFYALL)

    if attr_list[0] != "syft":
        raise ReferenceError(f"Reference don't match: {attr_list[0]}")

    if (
        attr_list[1] != "core"
        and attr_list[1] != "lib"
        and attr_list[1] != "grid"
        and attr_list[1] != "wrappers"
        and attr_list[1] != "proxy"
    ):
        raise ReferenceError(f"Reference don't match: {attr_list[1]}")

    return index_modules(a_dict=globals()["syft"], keys=attr_list[1:])


def get_fully_qualified_name(obj: object) -> str:
    """Return the full path and name of a class

    Sometimes we want to return the entire path and name encoded
    using periods. For example syft.core.common.message.SyftMessage
    is the current fully qualified path and name for the SyftMessage
    object.

    Args:
        obj: the object we want to get the name of

    Returns:
        the full path and name of the object

    """

    fqn = obj.__class__.__module__
    try:
        fqn += "." + obj.__class__.__name__
    except Exception as e:
        error(f"Failed to get FQN: {e}")
    return fqn


def aggressive_set_attr(obj: object, name: str, attr: object) -> None:
    """Different objects prefer different types of monkeypatching - try them all

    Args:
        obj: object whose attribute has to be set
        name: attribute name
        attr: value given to the attribute

    """
    try:
        setattr(obj, name, attr)
    except Exception:
        curse(obj, name, attr)


def obj2pointer_type(obj: object) -> type:
    fqn = None
    try:
        fqn = get_fully_qualified_name(obj=obj)
    except Exception as e:
        # sometimes the object doesn't have a __module__ so you need to use the type
        # like: collections.OrderedDict
        debug(f"Unable to get get_fully_qualified_name of {type(obj)} trying type. {e}")
        fqn = get_fully_qualified_name(obj=type(obj))

    # TODO: fix for other types
    if obj is None:
        fqn = "syft.lib.python._SyNone"

    try:
        ref = syft.lib_ast.query(fqn, obj_type=type(obj))
    except Exception as e:
        log = f"Cannot find {type(obj)} {fqn} in lib_ast. {e}"
        critical(log)
        raise Exception(log)

    return ref.pointer_type  # type: ignore


def key_emoji(key: object) -> str:
    try:
        if isinstance(key, (bytes, SigningKey, VerifyKey)):
            hex_chars = bytes(key).hex()[-8:]
            return char_emoji(hex_chars=hex_chars)
    except Exception as e:
        error(f"Fail to get key emoji: {e}")
        pass
    return "ALL"


def char_emoji(hex_chars: str) -> str:
    base = ord("\U0001F642")
    hex_base = ord("0")
    code = 0
    for char in hex_chars:
        offset = ord(char)
        code += offset - hex_base
    return chr(base + code)


left_name = [
    "admiring",
    "adoring",
    "affectionate",
    "agitated",
    "amazing",
    "angry",
    "awesome",
    "beautiful",
    "blissful",
    "bold",
    "boring",
    "brave",
    "busy",
    "charming",
    "clever",
    "cool",
    "compassionate",
    "competent",
    "condescending",
    "confident",
    "cranky",
    "crazy",
    "dazzling",
    "determined",
    "distracted",
    "dreamy",
    "eager",
    "eagleman",
    "ecstatic",
    "elastic",
    "elated",
    "elegant",
    "eloquent",
    "epic",
    "exciting",
    "fervent",
    "festive",
    "flamboyant",
    "focused",
    "friendly",
    "frosty",
    "funny",
    "gallant",
    "gifted",
    "goofy",
    "gracious",
    "great",
    "happy",
    "hardcore",
    "heuristic",
    "hopeful",
    "hungry",
    "infallible",
    "inspiring",
    "interesting",
    "intelligent",
    "jolly",
    "jovial",
    "keen",
    "kind",
    "laughing",
    "loving",
    "lucid",
    "magical",
    "mystifying",
    "modest",
    "musing",
    "naughty",
    "nervous",
    "nice",
    "nifty",
    "nostalgic",
    "objective",
    "optimistic",
    "peaceful",
    "pedantic",
    "pensive",
    "practical",
    "priceless",
    "quirky",
    "quizzical",
    "recursing",
    "relaxed",
    "reverent",
    "romantic",
    "sad",
    "serene",
    "sharp",
    "silly",
    "sleepy",
    "stoic",
    "strange",
    "stupefied",
    "suspicious",
    "sweet",
    "tender",
    "thirsty",
    "trusting",
    "unruffled",
    "upbeat",
    "vibrant",
    "vigilant",
    "vigorous",
    "wizardly",
    "wonderful",
    "xenodochial",
    "youthful",
    "zealous",
    "zen",
]

right_name = [
    "altman",
    "bach",
    "bengio",
    "bostrom",
    "botvinick",
    "brockman",
    "chintala",
    "chollet",
    "chomsky",
    "dean",
    "fridman",
    "gardner",
    "goertzel",
    "goodfellow",
    "hassabis",
    "he",
    "hinton",
    "hochreiter",
    "hotz",
    "howard",
    "hutter",
    "kaliouby",
    "karp",
    "karpathy",
    "kearns",
    "kellis",
    "knuth",
    "koller",
    "krizhevsky",
    "larochelle",
    "lecun",
    "li",
    "lim",
    "malik",
    "mironov",
    "ng",
    "norvig",
    "olah",
    "pearl",
    "pesenti",
    "russell",
    "salakhutdinov",
    "schmidhuber",
    "silver",
    "smola",
    "song",
    "sophia",
    "sutskever",
    "thomas",
    "thrun",
    "trask",
    "vapnik",
    "vaswani",
    "vinyals",
    "winston",
    "wolf",
    "wolfram",
]


def random_name() -> str:
    left_i = randbelow(len(left_name) - 1)
    right_i = randbelow(len(right_name) - 1)
    return f"{left_name[left_i].capitalize()} {right_name[right_i].capitalize()}"


def inherit_tags(
    attr_path_and_name: str,
    result: object,
    self_obj: Optional[object],
    args: Union[tuple, list],
    kwargs: dict,
) -> None:
    tags = []
    if self_obj is not None and hasattr(self_obj, "tags"):
        tags.extend(list(self_obj.tags))  # type: ignore

    for arg in args:
        if hasattr(arg, "tags"):
            tags.extend([tag for tag in arg.tags if tag not in tags])

    for arg in kwargs.values():
        if hasattr(arg, "tags"):
            tags.extend([tag for tag in arg.tags if tag not in tags])

    # only generate new tags if the result actually inherit some tags.
    if tags:
        tags.append(attr_path_and_name.split(".")[-1])
        result.tags = tags  # type: ignore


def get_root_data_path() -> Path:
    # get the PySyft / data directory to share datasets between notebooks
    # on Linux and MacOS the directory is: ~/.syft/data"
    # on Windows the directory is: C:/Users/$USER/.syft/data

    data_dir = Path.home() / ".syft" / "data"

    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def str_to_bool(bool_str: Optional[str]) -> bool:
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


def verify_tls() -> bool:
    return not str_to_bool(str(os.environ.get("IGNORE_TLS_ERRORS", "0")))


def ssl_test() -> bool:
    return len(os.environ.get("REQUESTS_CA_BUNDLE", "")) > 0
