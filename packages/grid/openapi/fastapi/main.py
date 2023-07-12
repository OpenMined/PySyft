# stdlib
from typing import Dict
from typing import List
from typing import Optional

# third party
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Blue Book", version="0.2.0")


class ResearchModel(BaseModel):
    name: str


api_state: Dict[int, ResearchModel] = {7: ResearchModel(name="Ava")}


@app.get("/", operation_id="home", summary="Home Page")
def read_root() -> Dict:
    return {"Caleb": "Smith"}


@app.get("/models/", operation_id="get_all", summary="Get all the Models")
def get_all() -> List[ResearchModel]:
    return list(api_state.values())


@app.get("/models/{model_id}", operation_id="get_model", summary="Get a Model by index")
def get_model(model_id: int) -> Optional[ResearchModel]:
    model = api_state.get(model_id, None)
    return model


@app.put("/models/{model_id}", operation_id="set_model", summary="Set a Model by index")
def set_model(model_id: int, model: ResearchModel) -> ResearchModel:
    api_state[model_id] = model
    return model
