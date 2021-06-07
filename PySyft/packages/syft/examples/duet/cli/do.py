# run this from the interactive command line like:
# python -i examples/duet/cli/do.py

# or you can run these commands manually in the python REPL

# syft absolute
import syft as sy

duet = sy.launch_duet(loopback=True)
print("DO Store", duet.store)
