from typing import Union
from sklearn.datasets import fetch_openml
from fastapi import FastAPI

app = FastAPI()

@app.get("/predict")
def read_root(data=""):
    print(data)
    X_adult, y_adult = fetch_openml("adult", version=2, return_X_y=True)

    # Remove redundant and non-feature columns
    X_adult = X_adult.drop(["education-num", "fnlwgt"], axis="columns")
    X_adult.dtypes
    
    # myIA.predict
    return X_adult






# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
#     return {"item_id": item_id, "q": q}