# stdlib
import asyncio
import logging
import random


class Actor:
    cooldown_period: int = 1

    def __init__(self, name):
        self.name = name
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)  # Set default logging level
        handler = logging.StreamHandler()  # Log to stdout
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        self.actions = [
            method
            for method in dir(self)
            if callable(getattr(self, method))
            and getattr(getattr(self, method), "is_action", False)
        ]
        # Call setup if defined
        if hasattr(self, "setup") and callable(self.setup):
            self.setup()

    async def run(self):
        try:
            while True:
                action = random.choice(self.actions)
                await getattr(self, action)()
                if isinstance(self.cooldown_period, int):
                    cooldown = self.cooldown_period
                elif isinstance(self.cooldown_period, tuple):
                    cooldown = random.randint(*self.cooldown_period)
                await asyncio.sleep(cooldown)
        finally:
            # Call teardown if defined
            if hasattr(self, "teardown") and callable(self.teardown):
                self.teardown()


def action(func):
    func.is_action = True
    return func
