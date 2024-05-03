import pandas as pd
from zipfile import ZipFile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf

app = FastAPI()

class Passenger(BaseModel):
    Pclass: int
    Name: str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket: str
    Fare: float
    Cabin: str
    Embarked: str

@app.post("/predict/")
async def predict_survival(passenger: Passenger):
    # Öncelikle gelen veriyi bir DataFrame'e dönüştürme
    data = pd.DataFrame([passenger.dict()])
    
    # Model dosyasının yolu
    model_path = "path/to/your/model/directory/model.h5"

    # Modeli yükleme
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Model loading failed")
    
    # Veriyi modele uygun hale getirme
    label_encoder = LabelEncoder()
    categorical_columns = ['Name', 'Sex', 'Cabin', 'Embarked', 'Ticket']
    for column in categorical_columns:
        data[column] = label_encoder.fit_transform(data[column])
    
    # Tahmin yapma
    try:
        prediction = model.predict(data.drop(columns=["Age"]))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    return {"survival_probability": prediction[0][0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
