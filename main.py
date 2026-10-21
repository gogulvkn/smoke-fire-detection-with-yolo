import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from PIL import Image
from ultralytics import YOLO

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
model = YOLO("best.pt")   # load once at startup

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.3):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    r = model.predict(img, conf=conf, imgsz=512)[0]
    return [
        {"class": r.names[int(b.cls)], "conf": round(float(b.conf), 3),
         "box": [round(v, 1) for v in b.xyxy[0].tolist()]}
        for b in r.boxes
    ]

@app.get("/", response_class=HTMLResponse)
def home():
    return open("index.html", encoding="utf-8").read()