# run this from the interactive command line like:
# python -i examples/duet/cli/ds.py

# or you can run these commands manually in the python REPL

# syft absolute
import syft as sy

duet = sy.join_duet(loopback=True)
print("DS Store", duet.store)
