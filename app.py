import uvicorn
from Bank_note import BankNote 
import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI
app = FastAPI()
pi=open("classifier.pkl","rb")
classifier=pickle.load(pi)

@app.post('/predict')
def predict(data:BankNote):
    variance=data.variance
    skewness=data.skewness
    curtosis=data.curtosis
    entropy=data.entropy
    a=classifier.predict([[variance,skewness,curtosis,entropy]])
    if(a[0]>0.5):
        a="fake note"
    else:
        a="real note"
    return{"prediction":a}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)