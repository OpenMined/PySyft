import logging
from memory_profiler import LogFile, profile
import sys

# create logger
logger = logging.getLogger('memory_profile_log')
logger.setLevel(logging.DEBUG)

# create file handler which logs even debug messages
fh = logging.FileHandler("memory_profile.log")
fh.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(fh)

# sys.stdout = LogFile('memory_profile_log', reportIncrementFlag=False)

@profile
def send_data(torch, duet, tensor_size):
    from time import sleep
    
    sleep(2)
    tensor = torch.rand(tensor_size)
    
    sleep(2)
    tensor_ptr = tensor.send(duet)
    
    sleep(2)
    return tensor_ptr
