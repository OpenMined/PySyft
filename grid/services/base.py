from threading import Thread
import json
import base64
from bitcoin import base58

class BaseService(object):
	
	def __init__(self,worker):
		self.worker = worker
		self.api = self.worker.api

	
