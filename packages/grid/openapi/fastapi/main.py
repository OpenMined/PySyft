# stdlib
from typing import Dict
from typing import List

# third party
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Blue Book", version="0.2.0")


class Item(BaseModel):
    name: str


api_state: Dict[int, Item] = {}


@app.get("/", operation_id="home", summary="Home Page")
def read_root() -> Dict:
    return {"Hello": "World"}


@app.get("/items/", operation_id="get_all", summary="Get all the Items")
def get_all() -> List[Item]:
    return list(api_state.values())


@app.get("/items/{item_id}", operation_id="get_item", summary="Get an Item by index")
def get_item(item_id: int) -> Dict:
    item = api_state.get(item_id, None)
    return {"item_": item}


@app.put("/items/{item_id}", operation_id="set_item", summary="Set an Item by index")
def set_item(item_id: int, item: Item) -> Item:
    api_state[item_id] = item
    return item
