# Smoke & Fire Detection (YOLO11)

Detects **smoke** and **fire** in photos and live camera frames. A YOLO11s model trained on the
[D-Fire dataset](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo) on Kaggle,
served by a FastAPI backend with a single-file web interface.

> Not a safety system. It can miss fires and mistake sunsets or orange lights for flames.
> Use it only alongside real smoke alarms.

## Project structure

```
main.py            FastAPI server (POST /detect, serves index.html)
index.html         Web interface: upload, paste, drag and drop, live camera
best.pt            Trained YOLO11s weights (class 0 = smoke, class 1 = fire)
requirements.txt   Python dependencies
docs/              Full tutorial (PDF): training on Kaggle and deployment
```

## Run it

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open http://localhost:8000. Run the command from this folder, because `main.py` loads
`best.pt` and `index.html` from the current directory. The camera works only on `https` or `localhost`.

## API

```bash
curl -X POST -F "file=@test.jpg" "http://localhost:8000/detect?conf=0.3"
# [{"class":"fire","conf":0.87,"box":[x1,y1,x2,y2]}]
```

`box` is in pixels. `conf` is the minimum confidence (default 0.3). Interactive docs are at `/docs`.

## Training

Trained on a Kaggle Tesla T4 with Ultralytics (YOLO11s, `imgsz=512`, `batch=32`, 3-hour cap).
The full step-by-step pipeline is in [`docs/yolo_smoke_fire_tutorial.pdf`](docs/yolo_smoke_fire_tutorial.pdf).

## Results (test split)

| Class | mAP50 | mAP50-95 |
|-------|-------|----------|
| Smoke |       |          |
| Fire  |       |          |
| All   |       |          |

## Dataset and license

The dataset is not included. Check the D-Fire dataset's license before any commercial use.
