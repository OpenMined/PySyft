# Custom Exception
# Purposefully raise a custom error to retry the task in celery worker.
# Whenever celery worker receives this Exception, it retries the action
# for a specified duration


class RetriableError(Exception):
    pass


class ObjectNotInStore(RetriableError):
    pass
