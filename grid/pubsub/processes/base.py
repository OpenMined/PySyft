
class BaseProcess(object):
	
	def __init__(self,worker):
		self.worker = worker
		self.api = self.worker.api

	def get_openmined_nodes(self):
		nodes = self.workerapi.pubsub_peers('openmined')['Strings']
		if(nodes is not None):
			return nodes
		else:
			return []

	def get_nodes(self):
		nodes =  self.worker.api.pubsub_peers()['Strings']
		if(nodes is not None):
			return nodes
		else:
			return []

	def publish(self, channel, message):
		if isinstance(message, dict) or isinstance(message, list):
			self.api.pubsub_pub(topic=channel, payload=json.dumps(message))
		else:
			self.api.pubsub_pub(topic=channel, payload=message)

	def listen_to_channel_sync(self, *args):
		"""
		Synchronous version of listen_to_channel
		"""

		self.listen_to_channel_impl(*args)

	def listen_to_channel(self, *args):
		"""
		Listens for IPFS pubsub sub messages asynchronously.

		This function will create the listener and call back your handler
		function on a new thread.
		"""
		t1 = Thread(target=self.listen_to_channel_impl, args=args)
		t1.start()

	def listen_to_channel_impl(self, channel, handle_message,
							   init_function=None, ignore_from_self=False):
		"""
		Do not call directly.  Use listen_to_channel or listen_to_channel_sync instead.
		"""

		first_proc = True

		if channel not in self.worker.subscribed_list:
			
			print(f"SUBSCRIBING TO {channel}")
			new_messages = self.api.pubsub_sub(topic=channel, stream=True)
			self.worker.subscribed_list.append(channel)

		else:
			print(f"ALREADY SUBSCRIBED TO {channel}")
			return

		for m in new_messages:
			if init_function is not None and first_proc:
				init_function()
				first_proc = False

			message = self.decode_message(m)
			if message is not None:
				fr = base58.encode(message['from'])

				if not ignore_from_self or fr != self.worker.id:
					out = handle_message(message)
					if(out is not None):
						return out
				else:
					print("ignored message from self")

				# if(message is not None):
				#    if handle_message is not None:
				#        out = handle_message(message)
				#        if (out is not None):
				#            return out
				#        else:
				#            return message
				#    else:
				#        return message

	def decode_message(self, encoded):
		if('from' in encoded):
			decoded = {}
			decoded['from'] = base64.standard_b64decode(encoded['from'])
			decoded['data'] = base64.standard_b64decode(
				encoded['data']).decode('ascii')
			decoded['seqno'] = base64.standard_b64decode(encoded['seqno'])
			decoded['topicIDs'] = encoded['topicIDs']
			decoded['encoded'] = encoded
			return decoded
		else:
			return None
