# third party
from simtester import Actor
from simtester import action


class DataScientist(Actor):
    cooldown_period = (1, 5)

    def setup(self):
        self.logger.info(f"{self.name}: Setup complete")

    def teardown(self):
        self.logger.info(f"{self.name}: Teardown complete")

    @action
    async def whoami(self):
        self.logger.info(f"Actor {self.name} is running whoami")
        # self.client.account()
