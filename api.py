from fastapi import FastAPI
import pickle 
from pydantic import BaseModel
app=FastAPI()
with open('predict/vector.pkl','rb') as f:
    vector=pickle.load(f)
with open('predict/model.pkl','rb') as f:
    model=pickle.load(f)

class Base_model(BaseModel):
    reviewText:str

@app.post('/predict')
def prediction(item:Base_model):
    vector_pre=vector.transform([item.reviewText])
    model_pre=model.predict(vector_pre)
    return ('review prediction:',int(model_pre[0]))
