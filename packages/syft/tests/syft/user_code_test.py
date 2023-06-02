# import syft as sy
# from syft.service.action.action_object import ActionObject


# @sy.syft_function(input_policy=sy.ExactMatch(),
#                     output_policy=sy.SingleExecutionExactOutput())
# def test_func():
#     return 1

# def test_user_code(root_domain_client, guest_client):


#     test_func()
#     x = guest_client.api.services.code.request_code_execution(test_func)

#     root_domain_client._api = None
#     message = root_domain_client.notifications[-1]
#     request = message.link
#     user_code = request.changes[0].link
#     result = user_code.unsafe_function()
#     request.accept_by_depositing_result(result)

#     # guest_domain_client._api = None
#     # _ = guest_domain_client.api

#     result = guest_client.api.services.code.test_func()
#     assert isinstance(result, ActionObject)

#     real_result = result.get_from(guest_client)
#     assert isinstance(real_result, int)
