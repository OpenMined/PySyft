
from ...serde.deserialize import _deserialize as deserialize
from ...serde.serializable import serializable
from .base_queue import AbstractMessageHandler
import sys

@serializable()
class CodeExecutionMessageHandler(AbstractMessageHandler):
    queue_name = "code_execution"
    
    @staticmethod
    def handle_message(message: bytes):
        
        from ...node.node import Node
        # Interactions with the node:
        #   * fetching code_object
        #   * fetching policies
        #   * updating output policy state
        #   * updating action store with the result
        print("QUEUE:", file=sys.stderr)
        
        
        # task_uid, promised_result_id, code_item, filtered_kwargs, worker_setrtings = deserialize(message, from_bytes=True)
        
        task_uid, code_item_id, kwargs, worker_settings = deserialize(message, from_bytes=True)
        print(task_uid, code_item_id, kwargs, worker_settings, file=sys.stderr)
        
        # worker.api.call(code_item.id, ...)
        try:
            worker = Node(
                id=worker_settings.id,
                name=worker_settings.name,
                signing_key=worker_settings.signing_key,
                document_store_config=worker_settings.document_store_config,
                action_store_config=worker_settings.action_store_config,
                blob_storage_config=worker_settings.blob_store_config,
                is_subprocess=True,
            )
        
            print("Worker:", worker, file=sys.stderr)
        except Exception as e:
            print(e, file=sys.stderr)
            
        print("Done", file=sys.stderr)
            
            