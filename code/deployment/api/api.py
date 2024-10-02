from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from catboost import CatBoostRegressor
import os
import uvicorn
import yaml


# Initialize FastAPI app
app = FastAPI()

# Load the CatBoost model (change path to your model file)
'''
config_file = 'configs\deploy.yaml' 
with open(config_file, 'r') as file:
    config = yaml.safe_load(file)
'''

MODEL_PATH = "/app/models/catboost_model.cbm"
if not os.path.exists(MODEL_PATH):
    raise Exception(f"Model file not found at {MODEL_PATH}")
    
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

# Define the input schema using Pydantic
class PredictionRequest(BaseModel):
    Company: str
    TypeName: str
    Inches: float
    Ram: int
    OS: str
    Weight: float
    Screen: str
    ScreenW: int
    ScreenH: int
    Touchscreen: str  # Yes or No
    IPSpanel: str  # Yes or No
    RetinaDisplay: str  # Yes or No
    CPU_company: str
    CPU_freq: float
    PrimaryStorage: int
    SecondaryStorage: int
    PrimaryStorageType: str  # SSD, Flash Storage, etc.
    SecondaryStorageType: str  # Yes or No
    GPU_company: str

# Convert categorical values (like Yes/No) to numeric before passing to the model
def preprocess_input(data: PredictionRequest):
    # Convert the input dictionary to a pandas DataFrame
    input_data = pd.DataFrame([data.model_dump()])
    
    return input_data

# Define the prediction endpoint
@app.post("/predict")
async def predict(data: PredictionRequest):

    """
    Predict endpoint: Expects JSON data that matches the PredictionRequest schema.
    """
    try:
        # Preprocess the input data
        input_data = preprocess_input(data)

        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        
        # Return prediction as a response
        return {"prediction": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
