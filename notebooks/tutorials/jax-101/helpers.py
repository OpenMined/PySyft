def review_request(message):
    request = message.link
    func = request.changes[0].link
    func_name = func.service_func_name
    print(func_name)
    print(func.raw_code)


def run_submitted_function(message):
    request = message.link
    func = request.changes[0].link
    user_func = func.unsafe_function
    real_result = user_func()
    print(real_result)
    return real_result


def accept_request(message, real_result):
    request = message.link
    request.approve()
    result = request.accept_by_depositing_result(real_result)
    print(result)
