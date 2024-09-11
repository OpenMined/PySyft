# stdlib
import asyncio
from asyncio.selector_events import BaseSelectorEventLoop
from collections import deque
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime
import functools
import hashlib
import inspect
from itertools import chain
from itertools import repeat
import json
import logging
import multiprocessing
import multiprocessing as mp
from multiprocessing import set_start_method
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock as LockBase
import operator
import os
from pathlib import Path
import platform
import random
import re
import reprlib
import secrets
from secrets import randbelow
import socket
import sys
from sys import getsizeof
import threading
import time
import types
from types import ModuleType
from typing import Any

# third party
from IPython.display import display
from forbiddenfruit import curse
from nacl.signing import SigningKey
from nacl.signing import VerifyKey
import nh3
import requests

# relative
from ..serde.serialize import _serialize as serialize

logger = logging.getLogger(__name__)

DATASETS_URL = "https://raw.githubusercontent.com/OpenMined/datasets/main"
PANDAS_DATA = f"{DATASETS_URL}/pandas_cookbook"


def get_env(key: str, default: Any | None = None) -> str | None:
    return os.environ.get(key, default)


def full_name_with_qualname(klass: type) -> str:
    """Returns the klass module name + klass qualname.

    Args:
        klass (type): The class whose fully qualified name is needed.

    Returns:
        str: The fully qualified name of the class.
    """
    try:
        if not hasattr(klass, "__module__"):
            return f"builtins.{get_qualname_for(klass)}"
        return f"{klass.__module__}.{get_qualname_for(klass)}"
    except Exception as e:
        # try name as backup
        logger.error(f"Failed to get FQN for: {klass} {type(klass)}", exc_info=e)
    return full_name_with_name(klass=klass)


def full_name_with_name(klass: type) -> str:
    """Returns the klass module name + klass name.

    Args:
        klass (type): The class whose fully qualified name is needed.

    Returns:
        str: The fully qualified name of the class.

    Raises:
        Exception: If there is an error while getting the fully qualified name.
    """
    try:
        if not hasattr(klass, "__module__"):
            return f"builtins.{get_name_for(klass)}"
        return f"{klass.__module__}.{get_name_for(klass)}"
    except Exception as e:
        logger.error(f"Failed to get FQN for: {klass} {type(klass)}", exc_info=e)
        raise e


def get_qualname_for(klass: type) -> str:
    """Get the qualified name of a class.

    Args:
        klass (type): The class to get the qualified name for.

    Returns:
        str: The qualified name of the class.
    """
    qualname = getattr(klass, "__qualname__", None) or getattr(klass, "__name__", None)
    if qualname is None:
        qualname = extract_name(klass)
    return qualname


def get_name_for(klass: type) -> str:
    """Get the name of a class.

    Args:
        klass (type): The class to get the name for.

    Returns:
        str: The name of the class.
    """
    klass_name = getattr(klass, "__name__", None)
    if klass_name is None:
        klass_name = extract_name(klass)
    return klass_name


def get_mb_size(data: Any, handlers: dict | None = None) -> float:
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    Otherwise, tries to read from the __slots__ or __dict__ of the object.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    Lightly modified from
    https://code.activestate.com/recipes/577504-compute-memory-footprint-of-an-object-and-its-cont/
    which is referenced in official sys.getsizeof documentation
    https://docs.python.org/3/library/sys.html#sys.getsizeof.

    Args:
        data (Any): The object to calculate the memory size for.
        handlers (dict | None): Custom handlers for additional types.

    Returns:
        float: The memory size of the object in MB.
    """

    def dict_handler(d: dict[Any, Any]) -> Iterator[Any]:
        return chain.from_iterable(d.items())

    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    if handlers:
        all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o: Any) -> int:
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))  # type: ignore
                break
        else:
            # no __slots__ *usually* means a __dict__, but some special builtin classes
            # (such as `type(None)`) have neither else, `o` has no attributes at all,
            # so sys.getsizeof() actually returned the correct value
            if not hasattr(o.__class__, "__slots__"):
                if hasattr(o, "__dict__"):
                    s += sizeof(o.__dict__)
            else:
                s += sum(
                    sizeof(getattr(o, x))
                    for x in o.__class__.__slots__
                    if hasattr(o, x)
                )
        return s

    return sizeof(data) / (1024.0 * 1024.0)


def get_mb_serialized_size(data: Any) -> float:
    """Get the size of a serialized object in MB.

    Args:
        data (Any): The object to be serialized and measured.

    Returns:
        float: The size of the serialized object in MB if successful.
    """
    try:
        serialized_data = serialize(data, to_bytes=True)
        return sys.getsizeof(serialized_data) / (1024 * 1024)
    except Exception as e:
        data_type = type(data)
        raise TypeError(
            f"Failed to serialize data of type '{data_type.__module__}.{data_type.__name__}'."
            f" Data type not supported. Detailed error: {e}"
        )


def extract_name(klass: type) -> str:
    """Extract the name of a class from its string representation.

    Args:
        klass (type): The class to extract the name from.

    Returns:
        str: The extracted name of the class.

    Raises:
        ValueError: If the class name could not be extracted.
    """
    name_regex = r".+class.+?([\w\._]+).+"
    regex2 = r"([\w\.]+)"
    matches = re.match(name_regex, str(klass))

    if matches is None:
        matches = re.match(regex2, str(klass))

    if matches:
        try:
            fqn: str = matches[1]
            if "." in fqn:
                return fqn.split(".")[-1]
            return fqn
        except Exception as e:
            logger.error(f"Failed to get klass name {klass}", exc_info=e)
            raise e
    else:
        raise ValueError(f"Failed to match regex for klass {klass}")


def validate_type(_object: object, _type: type, optional: bool = False) -> Any:
    """Validate that an object is of a certain type.

    Args:
        _object (object): The object to validate.
        _type (type): The type to validate against.
        optional (bool): Whether the object can be None.

    Returns:
        Any: The validated object.

    Raises:
        Exception: If the object is not of the expected type.
    """
    if isinstance(_object, _type) or (optional and (_object is None)):
        return _object

    raise Exception(f"Object {_object} should've been of type {_type}, not {_object}.")


def validate_field(_object: object, _field: str) -> Any:
    """Validate that an object has a certain field.

    Args:
        _object (object): The object to validate.
        _field (str): The field to validate.

    Returns:
        Any: The value of the field.

    Raises:
        Exception: If the field is not set on the object.
    """
    object = getattr(_object, _field, None)

    if object is not None:
        return object

    raise Exception(f"Object {_object} has no {_field} field set.")


def get_fully_qualified_name(obj: object) -> str:
    """Return the full path and name of a class.

    Sometimes we want to return the entire path and name encoded
    using periods.

    Args:
        obj (object): The object we want to get the name of.

    Returns:
        str: The full path and name of the object.
    """
    fqn = obj.__class__.__module__

    try:
        fqn += "." + obj.__class__.__name__
    except Exception as e:
        logger.error(f"Failed to get FQN: {e}")
    return fqn


def aggressive_set_attr(obj: object, name: str, attr: object) -> None:
    """Different objects prefer different types of monkeypatching - try them all.

    Args:
        obj (object): The object whose attribute has to be set.
        name (str): The attribute name.
        attr (object): The value given to the attribute.
    """
    try:
        setattr(obj, name, attr)
    except Exception:
        curse(obj, name, attr)


def key_emoji(key: object) -> str:
    """Generate an emoji representation of a key.

    Args:
        key (object): The key object.

    Returns:
        str: An emoji string representing the key.
    """
    try:
        if isinstance(key, bytes | SigningKey | VerifyKey):
            hex_chars = bytes(key).hex()[-8:]
            return char_emoji(hex_chars=hex_chars)
    except Exception as e:
        logger.error(f"Fail to get key emoji: {e}")
        pass
    return "ALL"


def char_emoji(hex_chars: str) -> str:
    """Generate an emoji based on a hexadecimal string.

    Args:
        hex_chars (str): The hexadecimal string.

    Returns:
        str: An emoji string generated from the hexadecimal string.
    """
    base = ord("\U0001f642")
    hex_base = ord("0")
    code = 0
    for char in hex_chars:
        offset = ord(char)
        code += offset - hex_base
    return chr(base + code)


def get_root_data_path() -> Path:
    """Get the root data path for storing datasets.

    Returns:
        Path: The root data path.
    """
    # get the PySyft / data directory to share datasets between notebooks
    # on Linux and MacOS the directory is: ~/.syft/data"
    # on Windows the directory is: C:/Users/$USER/.syft/data

    data_dir = Path.home() / ".syft" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def download_file(url: str, full_path: str | Path) -> Path | None:
    """Download a file from a URL.

    Args:
        url (str): The URL of the file to download.
        full_path (str | Path): The full path where the file should be saved.

    Returns:
        Path | None: The path to the downloaded file, or None if the download failed.
    """
    full_path = Path(full_path)
    if not full_path.exists():
        r = requests.get(url, allow_redirects=True, verify=verify_tls())  # nosec
        if not r.ok:
            print(f"Got {r.status_code} trying to download {url}")
            return None
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(r.content)
    return full_path


def verify_tls() -> bool:
    """Verify whether TLS should be used.

    Returns:
        bool: True if TLS should be used, False otherwise.
    """
    return not str_to_bool(str(os.environ.get("IGNORE_TLS_ERRORS", "0")))


def ssl_test() -> bool:
    """Check if SSL is properly configured.

    Returns:
        bool: True if SSL is configured, False otherwise.
    """
    return len(os.environ.get("REQUESTS_CA_BUNDLE", "")) > 0


def initializer(event_loop: BaseSelectorEventLoop | None = None) -> None:
    """Set the same event loop to other threads/processes.

    This is needed because there are new threads/processes started with
    the Executor and they do not have an event loop set.

    Args:
        event_loop (BaseSelectorEventLoop | None): The event loop to set.
    """
    if event_loop:
        asyncio.set_event_loop(event_loop)


def split_rows(rows: Sequence, cpu_count: int) -> list:
    """Split a sequence of rows into chunks for parallel processing.

    Args:
        rows (Sequence): The sequence of rows to split.
        cpu_count (int): The number of chunks to split into.

    Returns:
        list: A list of row chunks.
    """
    n = len(rows)
    a, b = divmod(n, cpu_count)
    start = 0
    output = []
    for i in range(cpu_count):
        end = start + a + (1 if b - i - 1 >= 0 else 0)
        output.append(rows[start:end])
        start = end
    return output


def list_sum(*inp_lst: list[Any]) -> Any:
    """Sum a list of lists element-wise.

    Args:
        *inp_lst (list[Any]): The list of lists to sum.

    Returns:
        Any: The sum of the lists.
    """
    s = inp_lst[0]
    for i in inp_lst[1:]:
        s = s + i
    return s


@contextmanager
def concurrency_override(count: int = 1) -> Iterator:
    """Context manager to override concurrency count.

    Args:
        count (int): The concurrency count to set. Defaults to 1.

    Yields:
        Iterator: A context manager.
    """
    # this only effects local code so its best to use in unit tests
    try:
        os.environ["FORCE_CONCURRENCY_COUNT"] = f"{count}"
        yield None
    finally:
        os.environ["FORCE_CONCURRENCY_COUNT"] = "0"


def print_process(  # type: ignore
    message: str,
    finish: EventClass,
    success: EventClass,
    lock: LockBase,
    refresh_rate: float = 0.1,
) -> None:
    """Print a dynamic process message that updates periodically.

    Args:
        message (str): The message to print.
        finish (EventClass): Event to signal the finish of the process.
        success (EventClass): Event to signal the success of the process.
        lock (LockBase): A lock to synchronize the print output.
        refresh_rate (float, optional): The refresh rate for updating the message. Defaults to 0.1.
    """
    with lock:
        while not finish.is_set():
            print(f"{bcolors.bold(message)} .", end="\r")
            time.sleep(refresh_rate)
            sys.stdout.flush()
            print(f"{bcolors.bold(message)} ..", end="\r")
            time.sleep(refresh_rate)
            sys.stdout.flush()
            print(f"{bcolors.bold(message)} ...", end="\r")
            time.sleep(refresh_rate)
            sys.stdout.flush()
        if success.is_set():
            print(f"{bcolors.success(message)}" + (" " * len(message)), end="\n")
        else:
            print(f"{bcolors.failure(message)}" + (" " * len(message)), end="\n")
        sys.stdout.flush()


def print_dynamic_log(
    message: str,
) -> tuple[EventClass, EventClass]:
    """Print a dynamic log message that will change its color when some process is done.

    Args:
        message (str): The message to be printed.

    Returns:
        tuple[EventClass, EventClass]: Tuple of events that can control the log print from outside this method.
    """
    finish = multiprocessing.Event()
    success = multiprocessing.Event()
    lock = multiprocessing.Lock()

    if os_name() == "macOS":
        # set start method to fork in case of MacOS
        set_start_method("fork", force=True)

    multiprocessing.Process(
        target=print_process, args=(message, finish, success, lock)
    ).start()
    return (finish, success)


def find_available_port(
    host: str, port: int | None = None, search: bool = False
) -> int:
    """Find an available port on the specified host.

    Args:
        host (str): The host to check for available ports.
        port (int | None): The port to check. Defaults to a random port.
        search (bool): Whether to search for the next available port if the given port is in use.

    Returns:
        int: The available port number.

    Raises:
        Exception: If the port is not available and search is False.
    """
    if port is None:
        port = random.randint(1500, 65000)  # nosec
    port_available = False
    while not port_available:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result_of_check = sock.connect_ex((host, port))

            if result_of_check != 0:
                port_available = True
                break
            else:
                if search:
                    port += 1
                else:
                    break
            sock.close()

        except Exception as e:
            logger.error(f"Failed to check port {port}. {e}")
    sock.close()

    if search is False and port_available is False:
        error = (
            f"{port} is in use, either free the port or "
            + f"try: {port}+ to auto search for a port"
        )
        raise Exception(error)
    return port


def get_random_available_port() -> int:
    """Retrieve a random available port number from the host OS.

    Returns:
        int: Available port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as soc:
        soc.bind(("localhost", 0))
        return soc.getsockname()[1]


def get_loaded_syft() -> ModuleType:
    """Get the loaded Syft module.

    Returns:
        ModuleType: The loaded Syft module.
    """
    return sys.modules[__name__.split(".")[0]]


def get_subclasses(obj_type: type) -> list[type]:
    """Recursively generate the list of all classes within the sub-tree of an object.

    As a paradigm in Syft, we often allow for something to be known about by another
    part of the codebase merely because it has subclassed a particular object. While
    this can be a bit "magicish" it also can simplify future extensions and reduce
    the likelihood of small mistakes (if done right).

    This is a utility function which allows us to look for sub-classes and the sub-classes
    of those sub-classes etc. returning a full list of descendants of a class in the inheritance
    hierarchy.

    Args:
        obj_type (type): The type we want to look for sub-classes of.

    Returns:
        list[type]: The list of subclasses of obj_type.
    """

    classes = []
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes


def index_modules(a_dict: object, keys: list[str]) -> object:
    """Recursively find a Syft module from its path.

    This is the recursive inner function of index_syft_by_module_name.
    See that method for a full description.

    Args:
        a_dict (object): A module we're traversing.
        keys (list[str]): The list of string attributes we're using to traverse the module.

    Returns:
        object: A reference to the final object.
    """
    if len(keys) == 0:
        return a_dict
    return index_modules(a_dict=a_dict.__dict__[keys[0]], keys=keys[1:])


def index_syft_by_module_name(fully_qualified_name: str) -> object:
    """Look up a Syft class/module/function from full path and name.

    Sometimes we want to use the fully qualified name (such as one
    generated from the 'get_fully_qualified_name' method below) to
    fetch an actual reference. This is most commonly used in deserialization
    so that we can have generic protobuf objects which just have a string
    representation of the specific object it is meant to deserialize to.

    Args:
        fully_qualified_name (str): The name in str of a module, class, or function.

    Returns:
        object: A reference to the actual object at that string path.

    Raises:
        ReferenceError: If the reference does not match expected patterns.
    """
    # @Tudor this needs fixing during the serde refactor
    # we should probably just support the native type names as lookups for serde
    if fully_qualified_name == "builtins.NoneType":
        fully_qualified_name = "syft.lib.python._SyNone"
    attr_list = fully_qualified_name.split(".")

    if attr_list[0] != "syft":
        raise ReferenceError(f"Reference don't match: {attr_list[0]}")

    return index_modules(a_dict=get_loaded_syft(), keys=attr_list[1:])


def obj2pointer_type(obj: object | None = None, fqn: str | None = None) -> type:
    """Get the pointer type for an object based on its fully qualified name.

    Args:
        obj (object | None): The object to get the pointer type for.
        fqn (str | None): The fully qualified name of the object.

    Returns:
        type: The pointer type for the object.

    Raises:
        Exception: If the pointer type cannot be found.
    """
    if fqn is None:
        try:
            fqn = get_fully_qualified_name(obj=obj)
        except Exception as e:
            # sometimes the object doesn't have a __module__ so you need to use the type
            # like: collections.OrderedDict
            logger.debug(
                f"Unable to get get_fully_qualified_name of {type(obj)} trying type. {e}"
            )
            fqn = get_fully_qualified_name(obj=type(obj))

        # TODO: fix for other types
        if obj is None:
            fqn = "syft.lib.python._SyNone"

    try:
        ref = get_loaded_syft().lib_ast.query(fqn, obj_type=type(obj))
    except Exception:
        raise Exception(f"Cannot find {type(obj)} {fqn} in lib_ast.")

    return ref.pointer_type


def prompt_warning_message(message: str, confirm: bool = False) -> bool:
    """Prompt a warning message and optionally request user confirmation.

    Args:
        message (str): The warning message to display.
        confirm (bool): Whether to request user confirmation.

    Returns:
        bool: True if the user confirms, False otherwise.
    """
    # relative
    from ..service.response import SyftWarning

    warning = SyftWarning(message=message)
    display(warning)

    while confirm:
        response = input("Would you like to proceed? [y/n]: ").lower()
        if response == "y":
            return True
        elif response == "n":
            print("Aborted.")
            return False
        else:
            print("Invalid response. Please enter Y or N.")

    return True


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
    "bengios",
    "bostrom",
    "botvinick",
    "brockman",
    "chintala",
    "chollet",
    "chomsky",
    "dean",
    "dolgov",
    "eckersley",
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
    "isbell",
    "kaliouby",
    "karp",
    "karpathy",
    "kearns",
    "kellis",
    "knuth",
    "koller",
    "krizhevsky",
    "larochelle",
    "lattner",
    "lecun",
    "li",
    "lim",
    "littman",
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
    """Generate a random name by combining a left and right name part.

    Returns:
        str: The generated random name.
    """
    left_i = randbelow(len(left_name) - 1)
    right_i = randbelow(len(right_name) - 1)
    return f"{left_name[left_i].capitalize()} {right_name[right_i].capitalize()}"


def inherit_tags(
    attr_path_and_name: str,
    result: object,
    self_obj: object | None,
    args: tuple[Any, ...] | list[Any],
    kwargs: dict[str, Any],
) -> None:
    """Inherit tags from input objects to the result object.

    Args:
        attr_path_and_name (str): The attribute path and name to add as a tag.
        result (object): The result object to inherit tags.
        self_obj (object | None): The object that might have tags.
        args (tuple[Any, ...] | list[Any]): Arguments that might have tags.
        kwargs (dict[str, Any]): Keyword arguments that might have tags.
    """
    tags = []
    if self_obj is not None and hasattr(self_obj, "tags"):
        tags.extend(list(self_obj.tags))

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


def autocache(
    url: str, extension: str | None = None, cache: bool = True
) -> Path | None:
    """Automatically cache a file from a URL.

    Args:
        url (str): The URL of the file to cache.
        extension (str | None): The file extension to use.
        cache (bool): Whether to use the cache if the file already exists.

    Returns:
        Path | None: The path to the cached file, or None if caching failed.
    """
    try:
        data_path = get_root_data_path()
        file_hash = hashlib.sha256(url.encode("utf8")).hexdigest()
        filename = file_hash
        if extension:
            filename += f".{extension}"
        file_path = data_path / filename
        if os.path.exists(file_path) and cache:
            return file_path
        return download_file(url, file_path)
    except Exception as e:
        print(f"Failed to autocache: {url}. {e}")
        return None


def str_to_bool(bool_str: str | None) -> bool:
    """Convert a string to a boolean value.

    Args:
        bool_str (str | None): The string to convert.

    Returns:
        bool: The converted boolean value.
    """
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


# local scope functions cant be pickled so this needs to be global
def parallel_execution(
    fn: Callable[..., Any],
    parties: list[Any] | None = None,
    cpu_bound: bool = False,
) -> Callable[..., list[Any]]:
    """Wrap a function such that it can be run in parallel at multiple parties.

    Args:
        fn (Callable[..., Any]): The function to run.
        parties (list[Any] | None): Clients from syft. If this is set, then the
            function should be run remotely. Defaults to None.
        cpu_bound (bool): Whether to use processes instead of threads.

    Returns:
        Callable[..., list[Any]]: A Callable that returns a list of results.
    """

    @functools.wraps(fn)
    def wrapper(
        args: list[list[Any]],
        kwargs: dict[Any, dict[Any, Any]] | None = None,
    ) -> list[Any]:
        """Wrap sanity checks and checks what executor should be used.

        Args:
            args (list[list[Any]]): The list of lists of arguments.
            kwargs (dict[Any, dict[Any, Any]] | None): The dictionary of keyword arguments.

        Returns:
            list[Any]: The list of results from the parties.

        Raises:
            Exception: If the arguments list is empty.
        """
        if args is None or len(args) == 0:
            raise Exception("Parallel execution requires more than 0 args")

        # _base.Executor
        executor: type
        if cpu_bound:
            executor = ProcessPoolExecutor
            # asyncio objects cannot pickled and sent across processes
            # AttributeError: Can't pickle local object 'WeakSet.__init__.<locals>._remove'
            loop = None
        else:
            executor = ThreadPoolExecutor
            loop = asyncio.get_event_loop()

        # Each party has a list of args and a dictionary of kwargs
        nr_parties = len(args)

        if kwargs is None:
            kwargs = {}

        if parties:
            func_name = f"{fn.__module__}.{fn.__qualname__}"
            attr_getter = operator.attrgetter(func_name)
            funcs = [attr_getter(party) for party in parties]
        else:
            funcs = list(repeat(fn, nr_parties))

        futures = []

        with executor(
            max_workers=nr_parties, initializer=initializer, initargs=(loop,)
        ) as executor:
            for i in range(nr_parties):
                _args = args[i]
                _kwargs = kwargs
                futures.append(executor.submit(funcs[i], *_args, **_kwargs))

        local_shares = [f.result() for f in futures]

        return local_shares

    return wrapper


def concurrency_count(factor: float = 0.8) -> int:
    """Get the current concurrency count based on CPU count and a factor.

    Args:
        factor (float): The factor to apply to the CPU count. Defaults to 0.8.

    Returns:
        int: The calculated concurrency count.
    """
    force_count = int(os.environ.get("FORCE_CONCURRENCY_COUNT", 0))
    mp_count = force_count if force_count >= 1 else int(mp.cpu_count() * factor)
    return mp_count


class bcolors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    BLACK = "\033[99m"

    @staticmethod
    def green(message: str) -> str:
        """Return a green-colored string.

        Args:
            message (str): The message to color.

        Returns:
            str: The green-colored message.
        """
        return bcolors.GREEN + message + bcolors.ENDC

    @staticmethod
    def red(message: str) -> str:
        """Return a red-colored string.

        Args:
            message (str): The message to color.

        Returns:
            str: The red-colored message.
        """
        return bcolors.RED + message + bcolors.ENDC

    @staticmethod
    def yellow(message: str) -> str:
        """Return a yellow-colored string.

        Args:
            message (str): The message to color.

        Returns:
            str: The yellow-colored message.
        """
        return bcolors.YELLOW + message + bcolors.ENDC

    @staticmethod
    def bold(message: str, end_color: bool = False) -> str:
        """Return a bold string.

        Args:
            message (str): The message to bold.
            end_color (bool): Whether to reset color after the message.

        Returns:
            str: The bolded message.
        """
        msg = bcolors.BOLD + message
        if end_color:
            msg += bcolors.ENDC
        return msg

    @staticmethod
    def underline(message: str, end_color: bool = False) -> str:
        """Return an underlined string.

        Args:
            message (str): The message to underline.
            end_color (bool): Whether to reset color after the message.

        Returns:
            str: The underlined message.
        """
        msg = bcolors.UNDERLINE + message
        if end_color:
            msg += bcolors.ENDC
        return msg

    @staticmethod
    def warning(message: str) -> str:
        """Return a warning-colored string.

        Args:
            message (str): The message to color.

        Returns:
            str: The warning-colored message.
        """
        return bcolors.bold(bcolors.yellow(message))

    @staticmethod
    def success(message: str) -> str:
        """Return a success-colored string.

        Args:
            message (str): The message to color.

        Returns:
            str: The success-colored message.
        """
        return bcolors.green(message)

    @staticmethod
    def failure(message: str) -> str:
        """Return a failure-colored string.

        Args:
            message (str): The message to color.

        Returns:
            str: The failure-colored message.
        """
        return bcolors.red(message)


def os_name() -> str:
    """Get the name of the operating system.

    Returns:
        str: The name of the operating system.
    """
    os_name = platform.system()
    if os_name.lower() == "darwin":
        return "macOS"
    else:
        return os_name


# Note: In the future there might be other interpreters that we want to use
def is_interpreter_jupyter() -> bool:
    """Check if the current interpreter is Jupyter.

    Returns:
        bool: True if the current interpreter is Jupyter, False otherwise.
    """
    return get_interpreter_module() == "ipykernel.zmqshell"


def is_interpreter_colab() -> bool:
    """Check if the current interpreter is Google Colab.

    Returns:
        bool: True if the current interpreter is Google Colab, False otherwise.
    """
    return get_interpreter_module() == "google.colab._shell"


def is_interpreter_standard() -> bool:
    """Check if the current interpreter is a standard Python interpreter.

    Returns:
        bool: True if the current interpreter is standard, False otherwise.
    """
    return get_interpreter_module() == "StandardInterpreter"


def get_interpreter_module() -> str:
    """Get the module name of the current interpreter.

    Returns:
        str: The module name of the current interpreter.
    """
    try:
        # third party
        from IPython import get_ipython

        shell = get_ipython().__class__.__module__
        return shell
    except NameError:
        return "StandardInterpreter"  # not sure


if os_name() == "macOS":
    # needed on MacOS to prevent [__NSCFConstantString initialize] may have been in
    # progress in another thread when fork() was called.
    multiprocessing.set_start_method("spawn", True)


def thread_ident() -> int | None:
    """Get the identifier of the current thread.

    Returns:
        int | None: The thread identifier, or None if not available.
    """
    return threading.current_thread().ident


def proc_id() -> int:
    """Get the process ID of the current process.

    Returns:
        int: The process ID.
    """
    return os.getpid()


def set_klass_module_to_syft(klass: type, module_name: str) -> None:
    """Set the module of a class to Syft.

    Args:
        klass (type): The class to set the module for.
        module_name (str): The name of the module.
    """
    if module_name not in sys.modules["syft"].__dict__:
        new_module = types.ModuleType(module_name)
    else:
        new_module = sys.modules["syft"].__dict__[module_name]
    setattr(new_module, klass.__name__, klass)
    sys.modules["syft"].__dict__[module_name] = new_module


def get_queue_address(port: int) -> str:
    """Get queue address based on container host name.

    Args:
        port (int): The port number.

    Returns:
        str: The queue address.
    """
    container_host = os.getenv("CONTAINER_HOST", None)
    if container_host == "k8s":
        return f"tcp://backend:{port}"
    elif container_host == "docker":
        return f"tcp://{socket.gethostname()}:{port}"
    return f"tcp://localhost:{port}"


def get_dev_mode() -> bool:
    """Check if the application is running in development mode.

    Returns:
        bool: True if in development mode, False otherwise.
    """
    return str_to_bool(os.getenv("DEV_MODE", "False"))


def generate_token() -> str:
    """Generate a secure random token.

    Returns:
        str: The generated token.
    """
    return secrets.token_hex(64)


def sanitize_html(html: str) -> str:
    """Sanitize HTML content by allowing specific tags and attributes.

    Args:
        html (str): The HTML content to sanitize.

    Returns:
        str: The sanitized HTML content.
    """
    policy = {
        "tags": ["svg", "strong", "rect", "path", "circle", "code", "pre"],
        "attributes": {
            "*": {"class", "style"},
            "svg": {
                "class",
                "style",
                "xmlns",
                "width",
                "height",
                "viewBox",
                "fill",
                "stroke",
                "stroke-width",
            },
            "path": {"d", "fill", "stroke", "stroke-width"},
            "rect": {"x", "y", "width", "height", "fill", "stroke", "stroke-width"},
            "circle": {"cx", "cy", "r", "fill", "stroke", "stroke-width"},
        },
        "remove": {"script", "style"},
    }

    tags = nh3.ALLOWED_TAGS
    for tag in policy["tags"]:
        tags.add(tag)

    _attributes = deepcopy(nh3.ALLOWED_ATTRIBUTES)
    attributes = {**_attributes, **policy["attributes"]}  # type: ignore

    return nh3.clean(
        html_str,
        tags=tags,
        clean_content_tags=policy["remove"],
        attributes=attributes,
    )


def parse_iso8601_date(date_string: str) -> datetime:
    """Parse an ISO8601 date string into a datetime object.

    Args:
        date_string (str): The ISO8601 date string.

    Returns:
        datetime: The parsed datetime object.
    """
    # Handle variable length of microseconds by trimming to 6 digits
    if "." in date_string:
        base_date, microseconds = date_string.split(".")
        microseconds = microseconds.rstrip("Z")  # Remove trailing 'Z'
        microseconds = microseconds[:6]  # Trim to 6 digits
        date_string = f"{base_date}.{microseconds}Z"
    return datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S.%fZ")


def get_latest_tag(registry: str, repo: str) -> str | None:
    """Get the latest tag from a Docker registry for a given repository.

    Args:
        registry (str): The Docker registry.
        repo (str): The repository name.

    Returns:
        str | None: The latest tag, or None if no tags are found.
    """
    repo_url = f"http://{registry}/v2/{repo}"
    res = requests.get(url=f"{repo_url}/tags/list", timeout=5)
    tags = res.json().get("tags", [])

    tag_times = []
    for tag in tags:
        manifest_response = requests.get(f"{repo_url}/manifests/{tag}", timeout=5)
        manifest = manifest_response.json()
        created_time = json.loads(manifest["history"][0]["v1Compatibility"])["created"]
        created_datetime = parse_iso8601_date(created_time)
        tag_times.append((tag, created_datetime))

    # sort tags by datetime
    tag_times.sort(key=lambda x: x[1], reverse=True)
    if len(tag_times) > 0:
        return tag_times[0][0]
    return None



def get_caller_file_path() -> str | None:
    stack = inspect.stack()

    for frame_info in stack:
        code_context = frame_info.code_context
        if code_context and len(code_context) > 0:
            if "from syft import test_settings" in str(frame_info.code_context):
                caller_file_path = os.path.dirname(os.path.abspath(frame_info.filename))
                return caller_file_path

    return None


def find_base_dir_with_tox_ini(start_path: str = ".") -> str | None:
    base_path = os.path.abspath(start_path)
    while True:
        if os.path.exists(os.path.join(base_path, "tox.ini")):
            return base_path
        parent_path = os.path.abspath(os.path.join(base_path, os.pardir))
        if parent_path == base_path:  # Reached the root directory
            break
        base_path = parent_path
    return start_path


def get_all_config_files(base_path: str, current_path: str) -> list[str]:
    config_files = []
    current_path = os.path.abspath(current_path)
    base_path = os.path.abspath(base_path)

    while current_path.startswith(base_path):
        config_file = os.path.join(current_path, "settings.yaml")
        if os.path.exists(config_file):
            config_files.append(config_file)
        if current_path == base_path:  # Stop if we reach the base directory
            break
        current_path = os.path.abspath(os.path.join(current_path, os.pardir))

    return config_files


def test_settings() -> Any:
    # third party
    from dynaconf import Dynaconf

    config_files = []
    current_path = "."

    # jupyter uses "." which resolves to the notebook
    if not is_interpreter_jupyter():
        # python uses the file which has from syft import test_settings in it
        import_path = get_caller_file_path()
        if import_path:
            current_path = import_path

    base_dir = find_base_dir_with_tox_ini(current_path)
    config_files = get_all_config_files(base_dir, current_path)
    config_files = list(reversed(config_files))
    # create
    # can override with
    # import os
    # os.environ["TEST_KEY"] = "var"
    # third party

    # Dynaconf settings
    test_settings = Dynaconf(
        settings_files=config_files,
        environments=True,
        envvar_prefix="TEST",
    )

    return test_settings


class CustomRepr(reprlib.Repr):
    def repr_str(self, obj: Any, level: int = 0) -> str:
        """Return a truncated string representation if it is too long.

        Args:
            obj (Any): The object to represent.
            level (int): The level of detail in the representation.

        Returns:
            str: The truncated string representation.
        """
        if len(obj) <= self.maxstring:
            return repr(obj)
        return repr(obj[: self.maxstring] + "...")


def repr_truncation(obj: Any, max_elements: int = 10) -> str:
    """Return a truncated string representation of the object if it is too long.

    Args:
        obj (Any): The object to be represented (can be str, list, dict, set...).
        max_elements (int): Maximum number of elements to display before truncating.

    Returns:
        str: A string representation of the object, truncated if necessary.
    """
    r = CustomRepr()
    r.maxlist = max_elements  # For lists
    r.maxdict = max_elements  # For dictionaries
    r.maxset = max_elements  # For sets
    r.maxstring = 100  # For strings
    r.maxother = 100  # For other objects

    return r.repr(obj)
