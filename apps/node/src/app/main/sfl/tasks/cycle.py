# Standard python imports
import logging
import traceback

# Local imports
from ... import executor


def run_task_once(name, func, *args):
    future = executor.futures._futures.get(name)
    logging.info("future: %s" % str(future))
    logging.info("futures count: %d" % len(executor.futures._futures))
    # prevent running multiple threads
    if future is None or future.done() is True:
        executor.futures.pop(name)
        try:
            executor.submit_stored(name, func, *args)
        except Exception as e:
            logging.error(
                "Failed to start new thread: %s %s" % (str(e), traceback.format_exc())
            )
    else:
        logging.warning(
            "Skipping %s execution because previous one is not finished" % name
        )


def complete_cycle(cycle_manager, cycle_id):
    logging.info("running complete_cycle")
    try:
        cycle_manager.complete_cycle(cycle_id)
        return True
    except Exception as e:
        logging.error(
            "Error in complete_cycle task: %s %s" % (str(e), traceback.format_exc())
        )
        return e
