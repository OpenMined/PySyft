# third party
from gevent import monkey

# 🟡 TODO 30: Move this to where we manage the different concurrency modes later
# make sure its stable in containers and other run targets
monkey.patch_all(thread=False)
