from fastapi import FastAPI
from pydantic import BaseModel
from personality import Behavior
import pickle
import uvicorn
pi=open("classifier.pkl","rb")
classifier=pickle.load(pi)
app=FastAPI()
@app.post('/predict')
def predict(data: Behavior):

    features = [[
        data.Time_spent_Alone,
        data.Stage_fear,
        data.Social_event_attendance,
        data.Going_outside,
        data.Drained_after_socializing,
        data.Friends_circle_size,
        data.Post_frequency
    ]]

    a = classifier.predict(features)
    if a[0]==0:
        a="introvert"
    else:
        a="extrovert"
    return {"prediction": a}
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)