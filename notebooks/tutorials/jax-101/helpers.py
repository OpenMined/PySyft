def review_request(message):
    request = message.link
    for change in request.changes:
        if not hasattr(change, "link"):
            continue
        func = change.link
        func_name = func.service_func_name
        print(func_name)
        print(func.raw_code)


def run_submitted_function(message):
    request = message.link
    for change in request.changes:
        if not hasattr(change, "link"):
            continue
        func = change.link
        user_func = func.unsafe_function
        real_result = user_func()
        print(real_result)
        return real_result


def accept_request(message, real_result):
    request = message.link
    request.approve()
    if real_result is not None:
        result = request.accept_by_depositing_result(real_result)
        print(result)
