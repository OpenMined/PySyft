# run this command line like:
# python examples/duet/cli/do_no_threads.py

# WARNING if this is run in REPL / Interactive it will block so you need to figure
# out a solution to that separately

# stdlib
import asyncio
import os

os.environ["SYFT_USE_EVENT_LOOP_THREAD"] = "0"
loop = asyncio.new_event_loop()
asyncio._set_running_loop(loop)

# syft absolute
import syft as sy

duet = sy.launch_duet(loopback=True)

print("DO Store", duet.store)
