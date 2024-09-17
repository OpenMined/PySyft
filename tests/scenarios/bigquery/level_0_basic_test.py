# @pytest.mark.asyncio
# async def test_level_2_basic_scenario(request):
#     ensure_package_installed("google-cloud-bigquery", "google.cloud.bigquery")
#     ensure_package_installed("db-dtypes", "db_dtypes")

#     scenario = Scenario(
#         name="test_create_apis_and_triage_requests",
#         events=[
#             EVENT_USER_ADMIN_CREATED,
#             EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
#             EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED,
#             EVENT_WORKER_POOL_CREATED,
#             EVENT_ALLOW_GUEST_SIGNUP_DISABLED,
#             EVENT_USERS_CREATED,
#             EVENT_USERS_CREATED_CHECKED,
#             EVENT_QUERY_ENDPOINT_CREATED,
#             EVENT_QUERY_ENDPOINT_CONFIGURED,
#             EVENT_SCHEMA_ENDPOINT_CREATED,
#             EVENT_SUBMIT_QUERY_ENDPOINT_CREATED,
#             EVENT_SUBMIT_QUERY_ENDPOINT_CONFIGURED,
#             EVENT_USERS_CAN_QUERY_MOCK,
#             EVENT_USERS_CAN_SUBMIT_QUERY,
#             EVENT_USERS_QUERY_NOT_READY,
#             EVENT_ADMIN_APPROVED_FIRST_REQUEST,
#             EVENT_USERS_CAN_GET_APPROVED_RESULT,
#         ],
#     )

