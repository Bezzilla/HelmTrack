from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from pathlib import Path
import io

app = FastAPI(title="HelmTrack API", description="Helmet violation detection API")

MODEL_PATH = Path(__file__).parent.parent / "model" / "best.pt"
model = YOLO(str(MODEL_PATH))

@app.get("/")
def root():
    return {"message": "HelmTrack API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)
    detections = []

    for result in results:
        for box in result.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": round(float(box.conf), 4),
                "bbox": box.xyxy[0].tolist()
            })

    violation = any(d["class"] == "NoHelmet" for d in detections)

    return JSONResponse({
        "violation_detected": violation,
        "detections": detections
    })
