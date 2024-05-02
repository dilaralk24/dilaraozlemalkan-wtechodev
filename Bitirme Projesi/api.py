# Gerekli kütüphaneler
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

# FastAPI uygulaması
app = FastAPI()

# Giriş verisi için model
class InputData(BaseModel):
    # Modelinizin giriş verilerine göre burayı düzenleyin
    Name: int
    Sex: int
    Ticket: int
    Fare: float
    Cabin: int
    Embarked: int
    Pclass: int
    SibSp: int
    Parch: int

# Modelin yüklenmesi
model = tf.keras.models.load_model("model.h5") # Model dosya yolu

# Tahmin rotası
@app.post("/predict")
async def predict(data: InputData):
    # Giriş verilerini numpy array'e dönüştürme
    input_data = np.array([[
        data.Name, data.Sex, data.Ticket, data.Fare, data.Cabin, data.Embarked, data.Pclass, data.SibSp, data.Parch
    ]])

    try:
        # Tahmin yapma
        prediction = model.predict(input_data)
        # Tahmini döndürme
        return {"prediction": prediction[0][0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))