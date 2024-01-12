from typing import Union

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root(data=""):
    print(data)
    
    # myIA.predict
    return{"Result": "Prediction"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}