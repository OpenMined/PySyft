import logging
from time import sleep
from zmq.log.handlers import PUBHandler

from greetings import hello

zmq_log_handler = PUBHandler('tcp://127.0.0.1:123456')
zmq_log_handler.setFormatter(logging.Formatter(fmt='{name} > {message}', style='{'))
zmq_log_handler.setFormatter(logging.Formatter(fmt='{name} #{lineno:>3} > {message}', style='{'), logging.DEBUG)
zmq_log_handler.setRootTopic('greeter')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(zmq_log_handler)

if __name__ == '__main__':
    sleep(0.1)
    msg_count = 5
    logger.warning('Preparing to greet the world...')
    for i in range(1,msg_count+1):
        logger.debug('Sending message {} of {}'.format(i,msg_count))
        hello.world()
        sleep(1.0)
    logger.info('Done!')
