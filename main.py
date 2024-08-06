import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from src.prediction import MakePredictions
from src.model import LoanDefaultPredictor
from src.preprocessing import DataPreprocessor
from fastapi.middleware.cors import CORSMiddleware
import sys

# make a new app
app = FastAPI()

# Directories
TRAIN_DIR = './data/train'
TEST_DIR = './data/test'
MODEL_DIR = './models'
SCALER_DIR = './data/scaler'
DATA_FILE_PATH = './data/Churn_Modelling.csv'


# Preprocessor instance
preprocessor = DataPreprocessor(file_path=DATA_FILE_PATH, scaler_dir=SCALER_DIR)

# Predictor instance
predictor = LoanDefaultPredictor(TRAIN_DIR, TEST_DIR, MODEL_DIR)

# Prediction instance
prediction_instance = MakePredictions(model_dir=MODEL_DIR, scaler_dir=SCALER_DIR)



class PredictionInput(BaseModel):
    '''
        Define the data needed to be passed through the API
    '''
    data: list

class TestSizeRandomState(BaseModel):
    '''
        Define the data needed to be passed through the API
    '''
    test_size: float
    random_state: int


@app.post("/preprocess/")
def preprocess_data(input: TestSizeRandomState):
    '''
        Preprocess the data using the defined preprocessing pipeline
    '''
    try:
        # Preprocess the data
        drop_columns = ["RowNumber", "CustomerId", "Surname", "Geography"]
        categorical_columns = ["Gender"]
        target_column = "Exited"
        X_train, X_test, y_train, y_test = preprocessor.preprocess(
            drop_columns=drop_columns,
            categorical_columns=categorical_columns,
            target_column=target_column,
            test_size=input.test_size if input.test_size else 0.2,
            random_state=input.random_state if input.random_state else 42,
        )
        preprocessor.save_datasets(X_train, X_test, y_train, y_test, TRAIN_DIR, TEST_DIR)
        preprocessor.save_scaler_used()
        return {
            "message": "Data preprocessed and saved successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rebuild-model/")
def rebuild_model():
    try:
        predictor.load_data()
        predictor.train_model()
        predictor.make_predictions()
        accuracy, report, matrix = predictor.evaluate_model()
        predictor.plot_confusion_matrix()
        predictor.plot_training_history()
        predictor.save_model()
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confusion-matrix/")
def get_confusion_matrix():
    try:
        return FileResponse('confusion_matrix.png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/training-history/")
def get_training_history():
    try:
        return FileResponse('training_history.png')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/")
def make_prediction(input: PredictionInput):
    try:
        # prediction_instance.load_scaler(SCALER_DIR)
        prediction_instance.load_model(model_number=-1)  # Load the latest model
        predictions = prediction_instance.make_prediction(input.data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

origins = [
    "http://localhost",
    "http://localhost:8080",
    "https://bankcustomerchurnpredictor.onrender.com"
    "http://localhost:3000",
    "https://churn-prediction-two.vercel.app",
]



app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
