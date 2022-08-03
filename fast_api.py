# Prerequisite: Install Uvicorn, OpenAPI
# To Run the File: uvicorn fast_api:app --reload

from fastapi import FastAPI, UploadFile, File
import shutil

# CV Libraries
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials

app = FastAPI()

@app.get("/")
async def welcome():
    return "Welcome to my Mini Project: Object Detection & Image Classification using Azure Custom Vision. Use: URL/docs to check the API Interface."

@app.post("/fd")
async def custom_vision_detector(file: UploadFile = File(...)):
   
    with open("testing/manual_test/temp1_cv.jpg", "wb") as f:
        shutil.copyfileobj(file.file, f)

    ENDPOINT = "Enter the Endpoint of Prediction"
    prediction_key = "Enter Your Prediction Key"
    project_id = "Enter You Project ID"
    publish_iteration_name = "Enter The Published Iteration Name"
    
    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)
    
    tags = []
    prob = []
    with open("testing/manual_test/temp1_cv.jpg", "rb") as image_contents:
        results = predictor.detect_image(project_id, publish_iteration_name, image_contents.read())
        for prediction in results.predictions:
            tags.append(prediction.tag_name)
            prob.append(prediction.probability * 100)

    return tuple(map(list, zip(tags, prob)))


@app.post("/sc")
async def custom_vision_classifier(file: UploadFile = File(...)):
   
    with open("testing/manual_test/temp2_cv.jpg", "wb") as f:
        shutil.copyfileobj(file.file, f)

    ENDPOINT = "Enter the Endpoint of Prediction"
    prediction_key = "Enter Your Prediction Key"
    project_id = "Enter You Project ID"
    publish_iteration_name = "Enter The Published Iteration Name"

    prediction_credentials = ApiKeyCredentials(in_headers = {"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    with open("testing/manual_test/temp2_cv.jpg", "rb") as image_contents:
        results = predictor.classify_image(project_id, publish_iteration_name, image_contents.read())
        tags = []
        prob = []
        for prediction in results.predictions:
            tags.append(prediction.tag_name)
            prob.append(prediction.probability * 100)

    return tuple(map(list, zip(tags, prob)))