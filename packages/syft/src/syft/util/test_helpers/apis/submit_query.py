# syft absolute
import syft as sy


def make_submit_query(settings, worker_pool_name):
    updated_settings = {"user_code_worker": worker_pool_name} | settings

    @sy.api_endpoint(
        path="bigquery.submit_query",
        description="API endpoint that allows you to submit SQL queries to run on the private data.",
        worker_pool_name=worker_pool_name,
        settings=updated_settings,
    )
    def submit_query(
        context,
        func_name: str,
        query: str,
    ) -> str:
        # syft absolute
        import syft as sy

        @sy.syft_function(
            name=func_name,
            input_policy=sy.MixedInputPolicy(
                endpoint=sy.Constant(
                    val=context.user_client.api.services.bigquery.test_query
                ),
                query=sy.Constant(val=query),
                client=context.user_client,
            ),
            worker_pool_name=context.settings["user_code_worker"],
        )
        def execute_query(query: str, endpoint):
            res = endpoint(sql_query=query)
            return res

        request = context.user_client.code.request_code_execution(execute_query)
        context.admin_client.requests.set_tags(request, ["autosync"])

        return f"Query submitted {request}. Use `client.code.{func_name}()` to run your query"

    return submit_query
