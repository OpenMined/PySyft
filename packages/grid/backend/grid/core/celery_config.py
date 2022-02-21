worker_send_task_event = False
task_ignore_result = True
task_time_limit = 600  # Rasswanth: should modify after optimizing PC
task_acks_late = True
broker_pool_limit = 500
worker_prefetch_multiplier = 1
task_routes = {
    "grid.worker.msg_without_reply": "main-queue",
    "delivery_mode": "transient",
}
accept_content = ["application/syft"]
task_serializer = "syft"
result_serializer = "syft"
