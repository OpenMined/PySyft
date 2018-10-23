from datetime import datetime
import cProfile
import pstats
from functools import wraps

PROFILE_MODE = True

SEND_MSG_STATS_LOG = "send_msg_profiling.log"
LOGFILE_LINE_FORMAT = "{}\tFrom: {}\tTo: {}\ttype: {}\t{:.2f} ms\ttotal calls: {}\n"

# how many milliseconds there are in one second
MS_IN_S = 1000


def save_send_msg_stats(prof, *args, **kwargs):
    """
    Saves _send_msg profiling information in SEND_MSG_STATS_LOG file.
    """

    sender = kwargs.get("self", args[0])
    encoded_message_wrapper = kwargs.get("message_json", args[1])
    recipient = kwargs.get("recipient", args[2])

    timestamp = datetime.now().strftime("%Y-%m-%d %T")
    message_wrapper = sender.decode_msg(encoded_message_wrapper)
    stats = pstats.Stats(prof)

    new_logfile_line = LOGFILE_LINE_FORMAT.format(
        timestamp,
        sender.id,
        recipient.id,
        message_wrapper["type"],
        stats.total_tt * MS_IN_S,
        stats.total_calls,
    )

    with open(SEND_MSG_STATS_LOG, "a+") as logfile:
        logfile.write(new_logfile_line)


def profile(profile_processing_fun):
    """
    Creates a profiling decorator for a method.

    :Parameters:

    * **profile_processing_fun (function)** a function which takes
      a cProfile.Profile object and the decorated method's arguments.
      The intended use is to have the function extract useful information
      and present it or store it somewhere (e.g. in a log).
    """

    def decorator(mthd):
        @wraps(mthd)
        def wrapper(*args, **kwargs):
            prof = cProfile.Profile()
            prof.enable()
            res = mthd(*args, **kwargs)
            prof.disable()
            profile_processing_fun(prof, *args, **kwargs)
            return res

        return wrapper

    return decorator
