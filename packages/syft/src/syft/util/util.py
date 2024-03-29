# stdlib
import asyncio
from asyncio.selector_events import BaseSelectorEventLoop
from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import functools
import hashlib
from itertools import repeat
import multiprocessing
import multiprocessing as mp
from multiprocessing import set_start_method
from multiprocessing.synchronize import Event as EventClass
from multiprocessing.synchronize import Lock as LockBase
import operator
import os
from pathlib import Path
import platform
import re
from secrets import randbelow
import socket
import sys
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
import requests

# relative
from .logger import critical
from .logger import debug
from .logger import error
from .logger import traceback_and_raise

DATASETS_URL = "https://raw.githubusercontent.com/OpenMined/datasets/main"
PANDAS_DATA = f"{DATASETS_URL}/pandas_cookbook"


def get_env(key: str, default: Any | None = None) -> str | None:
    return os.environ.get(key, default)


def full_name_with_qualname(klass: type) -> str:
    """Returns the klass module name + klass qualname."""
    try:
        if not hasattr(klass, "__module__"):
            return f"builtins.{get_qualname_for(klass)}"
        return f"{klass.__module__}.{get_qualname_for(klass)}"
    except Exception:
        # try name as backup
        print("Failed to get FQN for:", klass, type(klass))
    return full_name_with_name(klass=klass)


def full_name_with_name(klass: type) -> str:
    """Returns the klass module name + klass name."""
    try:
        if not hasattr(klass, "__module__"):
            return f"builtins.{get_name_for(klass)}"
        return f"{klass.__module__}.{get_name_for(klass)}"
    except Exception as e:
        print("Failed to get FQN for:", klass, type(klass))
        raise e


def get_qualname_for(klass: type) -> str:
    qualname = getattr(klass, "__qualname__", None) or getattr(klass, "__name__", None)
    if qualname is None:
        qualname = extract_name(klass)
    return qualname


def get_name_for(klass: type) -> str:
    klass_name = getattr(klass, "__name__", None)
    if klass_name is None:
        klass_name = extract_name(klass)
    return klass_name


def get_mb_size(data: Any) -> float:
    return sys.getsizeof(data) / (1024 * 1024)


def extract_name(klass: type) -> str:
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
            print(f"Failed to get klass name {klass}")
            raise e
    else:
        raise ValueError(f"Failed to match regex for klass {klass}")


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


def get_fully_qualified_name(obj: object) -> str:
    """Return the full path and name of a class

    Sometimes we want to return the entire path and name encoded
    using periods.

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


def key_emoji(key: object) -> str:
    try:
        if isinstance(key, bytes | SigningKey | VerifyKey):
            hex_chars = bytes(key).hex()[-8:]
            return char_emoji(hex_chars=hex_chars)
    except Exception as e:
        error(f"Fail to get key emoji: {e}")
        pass
    return "ALL"


def char_emoji(hex_chars: str) -> str:
    base = ord("\U0001f642")
    hex_base = ord("0")
    code = 0
    for char in hex_chars:
        offset = ord(char)
        code += offset - hex_base
    return chr(base + code)


def get_root_data_path() -> Path:
    # get the PySyft / data directory to share datasets between notebooks
    # on Linux and MacOS the directory is: ~/.syft/data"
    # on Windows the directory is: C:/Users/$USER/.syft/data

    data_dir = Path.home() / ".syft" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return data_dir


def download_file(url: str, full_path: str | Path) -> Path | None:
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
    return not str_to_bool(str(os.environ.get("IGNORE_TLS_ERRORS", "0")))


def ssl_test() -> bool:
    return len(os.environ.get("REQUESTS_CA_BUNDLE", "")) > 0


def initializer(event_loop: BaseSelectorEventLoop | None = None) -> None:
    """Set the same event loop to other threads/processes.
    This is needed because there are new threads/processes started with
    the Executor and they do not have have an event loop set
    Args:
        event_loop: The event loop.
    """
    if event_loop:
        asyncio.set_event_loop(event_loop)


def split_rows(rows: Sequence, cpu_count: int) -> list:
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
    s = inp_lst[0]
    for i in inp_lst[1:]:
        s = s + i
    return s


@contextmanager
def concurrency_override(count: int = 1) -> Iterator:
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
    refresh_rate=0.1,
) -> None:
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
    """
    Prints a dynamic log message that will change its color (to green or red) when some process is done.

    message: str = Message to be printed.

    return: tuple of events that can control the log print from the outside of this method.
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


def find_available_port(host: str, port: int, search: bool = False) -> int:
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

        except Exception as e:
            print(f"Failed to check port {port}. {e}")
    sock.close()

    if search is False and port_available is False:
        error = (
            f"{port} is in use, either free the port or "
            + f"try: {port}+ to auto search for a port"
        )
        raise Exception(error)
    return port


def get_loaded_syft() -> ModuleType:
    return sys.modules[__name__.split(".")[0]]


def get_subclasses(obj_type: type) -> list[type]:
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

    classes = []
    for sc in obj_type.__subclasses__():
        classes.append(sc)
        classes += get_subclasses(obj_type=sc)
    return classes


def index_modules(a_dict: object, keys: list[str]) -> object:
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

    if attr_list[0] != "syft":
        raise ReferenceError(f"Reference don't match: {attr_list[0]}")

    # if attr_list[1] != "core" and attr_list[1] != "user":
    #     raise ReferenceError(f"Reference don't match: {attr_list[1]}")

    return index_modules(a_dict=get_loaded_syft(), keys=attr_list[1:])


def obj2pointer_type(obj: object | None = None, fqn: str | None = None) -> type:
    if fqn is None:
        try:
            fqn = get_fully_qualified_name(obj=obj)
        except Exception as e:
            # sometimes the object doesn't have a __module__ so you need to use the type
            # like: collections.OrderedDict
            debug(
                f"Unable to get get_fully_qualified_name of {type(obj)} trying type. {e}"
            )
            fqn = get_fully_qualified_name(obj=type(obj))

        # TODO: fix for other types
        if obj is None:
            fqn = "syft.lib.python._SyNone"

    try:
        ref = get_loaded_syft().lib_ast.query(fqn, obj_type=type(obj))
    except Exception as e:
        log = f"Cannot find {type(obj)} {fqn} in lib_ast. {e}"
        critical(log)
        raise Exception(log)

    return ref.pointer_type


def prompt_warning_message(message: str, confirm: bool = False) -> bool:
    # relative
    from ..service.response import SyftWarning

    warning = SyftWarning(message=message)
    display(warning)

    while confirm:
        response = input("Would you like to proceed? [y/n]: ").lower()
        if response == "y":
            return True
        elif response == "n":
            display("Aborted !!")
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
    left_i = randbelow(len(left_name) - 1)
    right_i = randbelow(len(right_name) - 1)
    return f"{left_name[left_i].capitalize()} {right_name[right_i].capitalize()}"


def inherit_tags(
    attr_path_and_name: str,
    result: object,
    self_obj: object | None,
    args: tuple | list,
    kwargs: dict,
) -> None:
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
    result = False
    bool_str = str(bool_str).lower()
    if bool_str == "true" or bool_str == "1":
        result = True
    return result


# local scope functions cant be pickled so this needs to be global
def parallel_execution(
    fn: Callable[..., Any],
    parties: None | list[Any] = None,
    cpu_bound: bool = False,
) -> Callable[..., list[Any]]:
    """Wrap a function such that it can be run in parallel at multiple parties.
    Args:
        fn (Callable): The function to run.
        parties (Union[None, List[Any]]): Clients from syft. If this is set, then the
            function should be run remotely. Defaults to None.
        cpu_bound (bool): Because of the GIL (global interpreter lock) sometimes
            it makes more sense to use processes than threads if it is set then
            processes should be used since they really run in parallel if not then
            it makes sense to use threads since there is no bottleneck on the CPU side
    Returns:
        Callable[..., List[Any]]: A Callable that returns a list of results.
    """

    @functools.wraps(fn)
    def wrapper(
        args: list[list[Any]],
        kwargs: dict[Any, dict[Any, Any]] | None = None,
    ) -> list[Any]:
        """Wrap sanity checks and checks what executor should be used.
        Args:
            args (List[List[Any]]): Args.
            kwargs (Optional[Dict[Any, Dict[Any, Any]]]): Kwargs. Default to None.
        Returns:
            List[Any]: Results from the parties
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
        return bcolors.GREEN + message + bcolors.ENDC

    @staticmethod
    def red(message: str) -> str:
        return bcolors.RED + message + bcolors.ENDC

    @staticmethod
    def yellow(message: str) -> str:
        return bcolors.YELLOW + message + bcolors.ENDC

    @staticmethod
    def bold(message: str, end_color: bool = False) -> str:
        msg = bcolors.BOLD + message
        if end_color:
            msg += bcolors.ENDC
        return msg

    @staticmethod
    def underline(message: str, end_color: bool = False) -> str:
        msg = bcolors.UNDERLINE + message
        if end_color:
            msg += bcolors.ENDC
        return msg

    @staticmethod
    def warning(message: str) -> str:
        return bcolors.bold(bcolors.yellow(message))

    @staticmethod
    def success(message: str) -> str:
        return bcolors.green(message)

    @staticmethod
    def failure(message: str) -> str:
        return bcolors.red(message)


def os_name() -> str:
    os_name = platform.system()
    if os_name.lower() == "darwin":
        return "macOS"
    else:
        return os_name


# Note: In the future there might be other interpreters that we want to use
def is_interpreter_jupyter() -> bool:
    return get_interpreter_module() == "ipykernel.zmqshell"


def is_interpreter_colab() -> bool:
    return get_interpreter_module() == "google.colab._shell"


def is_interpreter_standard() -> bool:
    return get_interpreter_module() == "StandardInterpreter"


def get_interpreter_module() -> str:
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
    return threading.current_thread().ident


def proc_id() -> int:
    return os.getpid()


def set_klass_module_to_syft(klass: type, module_name: str) -> None:
    if module_name not in sys.modules["syft"].__dict__:
        new_module = types.ModuleType(module_name)
    else:
        new_module = sys.modules["syft"].__dict__[module_name]
    setattr(new_module, klass.__name__, klass)
    sys.modules["syft"].__dict__[module_name] = new_module


def get_syft_src_path() -> Path:
    return Path(__file__).parent.parent.parent.expanduser()


def get_grid_src_path() -> Path:
    syft_path = get_syft_src_path()
    return syft_path.parent.parent / "grid"


def get_syft_cpu_dockerfile() -> Path:
    return get_grid_src_path() / "backend" / "worker_cpu.dockerfile"


def get_queue_address(port: int) -> str:
    """Get queue address based on container host name."""

    container_host = os.getenv("CONTAINER_HOST", None)
    if container_host == "k8s":
        return f"tcp://backend:{port}"
    elif container_host == "docker":
        return f"tcp://{socket.gethostname()}:{port}"
    return f"tcp://localhost:{port}"


def get_dev_mode() -> bool:
    return str_to_bool(os.getenv("DEV_MODE", "False"))
