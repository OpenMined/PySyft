worker_send_task_event = False
task_ignore_result = True
task_time_limit = 300  # Rasswanth: should modify after optimizing PC
task_acks_late = True
broker_pool_limit = 500
worker_prefetch_multiplier = 1

# celery_app.conf.result_backend = "db+sqlite:///results.db"
# celery_app.conf.result_backend = "file:///tmp/results"
# celery_app.conf.result_backend = "postgresql://postgres:changethis@docker-host:5432"
# celery_app.conf.result_backend = "amqp://guest@queue//"
# celery_app.conf.result_backend = "rpc://"
# celery_app.conf.result_persistent = True
