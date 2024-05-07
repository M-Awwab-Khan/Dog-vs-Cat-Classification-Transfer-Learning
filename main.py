from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from utilities import classify_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

@app.post('/predict-image/')
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    predicted_class, confidence = classify_image(contents)
    return {
        "class": predicted_class,
        "confidence": confidence
    }
