from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Modelinizi yükleyin
model = load_model("model.h5")

# FastAPI uygulamasını oluşturun
app = FastAPI()

# Gelen veri için giriş modeli
class Item(BaseModel):
    Pclass: int
    Name:str
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Ticket:str
    Fare: float
    Cabin:str
    Embarked: str

# Tahmin endpoint'i
@app.post("/predict/")
async def predict_age(item: Item):
    # Gelen veriyi diziye dönüştürme
    input_data = np.array([[item.Pclass, item.Name, item.Sex, item.Age, item.SibSp, item.Parch, item.Ticket, item.Fare, item.Cabin, item.Embarked,]])

    # Tahmin yapma
    prediction = model.predict(input_data)

    # Tahmini dön
    return {"predicted_age": prediction[0][0]}

