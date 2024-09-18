# stdlib
import asyncio

# relative
from .actor import Actor


class Runner:
    def __init__(self, actor_classes: list[tuple[type[Actor], int]]):
        # `actor_classes` is a list of tuples (ActorClass, count)
        self.actor_classes = actor_classes

    async def start(self):
        tasks = []
        for actor_class, count in self.actor_classes:
            for i in range(count):
                # Instantiate the actor
                actor = actor_class(name=f"{actor_class.__name__}-{i}")
                tasks.append(asyncio.create_task(actor.run()))
        await asyncio.gather(*tasks)
