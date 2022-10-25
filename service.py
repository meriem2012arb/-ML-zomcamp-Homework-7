from pyexpat import model
import bentoml 
from pydantic import BaseModel 
from bentoml.io import NumpyNdarray
import numpy as np


#Q3 : pydantic
class UserProfile(BaseModel) :
    name : str
    age : int
    country : str
    rating : float

model1= 'mlzoomcamp_homework:qtzdz3slg6mwwdu5'
model2= 'mlzoomcamp_homework:jsi67fslz6txydu5'


runner = bentoml.sklearn.get(model1).to_runner()

svc = bentoml.Service("mlzoomcamp_homework", runners=[runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
async def classify(vector: np.ndarray) -> np.ndarray:
    result = await runner.predict.async_run (vector)
    return result

