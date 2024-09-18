# third party
from simtester import Actor
from simtester import action


class DataScientist(Actor):
    cooldown_period = (1, 5)

    def setup(self):
        # TODO create a server, or connect to an existing one
        # TODO register this test user on the server
        self.logger.info(f"{self.name}: Setup complete")

    def teardown(self):
        # TODO In case of live server, delete the user and perform any cleanup
        self.logger.info(f"{self.name}: Teardown complete")

    @action
    async def whoami(self):
        self.logger.info(f"Actor {self.name} is running whoami")
        # self.client.account()

    @action
    async def guest_register(self):
        # TODO register a guest user
        self.logger.info(f"Actor {self.name} is running guest_register")

    @action
    async def query_mock(self):
        # TODO run a query on bigquery.test_query
        self.logger.info(f"Actor {self.name} is running query_mock")

    @action
    async def submit_query(self):
        # TODO submit a query to bigquery.submit_query
        self.logger.info(f"Actor {self.name} is running submit_query")

    @action
    async def get_query_results(self):
        # TODO get the results of a query using get_results
        self.logger.info(f"Actor {self.name} is running get_query_results")


class Admin(Actor):
    cooldown_period = (1, 5)

    def setup(self):
        # TODO create a server, or connect to an existing one
        # server = make_server()
        # admin = make_admin()
        # self.root_client = admin.client(server)
        # create_prebuilt_worker_image
        # get_prebuilt_worker_image
        # add_external_registry
        self.logger.info(f"{self.name}: Setup complete")

    def teardown(self):
        self.logger.info(f"{self.name}: Teardown complete")

    @action
    async def approve_pending_requests(self):
        self.logger.info(f"Actor {self.name} is running approve_pending_requests")

    @action
    async def reject_pending_requests(self):
        self.logger.info(f"Actor {self.name} is running reject_pending_requests")

    @action
    async def create_worker_pool(self):
        self.logger.info(f"Actor {self.name} is running create_worker_pool")

    @action
    async def remove_worker_pool(self):
        self.logger.info(f"Actor {self.name} is running remove_worker_pool")

    @action
    async def disallow_guest_signup(self):
        self.logger.info(f"Actor {self.name} is running disallow_guest_signup")

    @action
    async def allow_guest_signup(self):
        self.logger.info(f"Actor {self.name} is running allow_guest_signup")

    @action
    async def create_endpoints(self):
        # create_endpoints_query
        # create_endpoints_schema
        # create_endpoints_submit_query
        self.logger.info(f"Actor {self.name} is running create_endpoints_query")

    @action
    async def remove_endpoints(self):
        # remove_endpoints_query
        # remove_endpoints_schema
        # remove_endpoints_submit_query
        self.logger.info(f"Actor {self.name} is running remove_endpoints_query")
