# run this command line like:
# python examples/duet/cli/ds_no_threads.py

# WARNING if this is run in REPL / Interactive it will block so you need to figure
# out a solution to that separately

# stdlib
import asyncio
import os

# syft absolute
import syft as sy

os.environ["SYFT_USE_EVENT_LOOP_THREAD"] = "0"
loop = asyncio.new_event_loop()
asyncio._set_running_loop(loop)


duet = sy.join_duet(loopback=True)

print("DS Store", duet.store)
