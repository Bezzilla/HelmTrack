# HelmTrack

AI-powered motorcycle helmet violation detection system using YOLOv8.

## Team
- Peerapat Seenoi
- Tanapoom Srikaewkheaw

## Project Overview
HelmTrack detects motorcycle riders without helmets from traffic images using YOLOv8 object detection. The system identifies 4 classes: Helmet, NoHelmet, Motorbike, and PNumber (license plate). A FastAPI backend serves the model and a Streamlit UI allows users to upload images and receive instant violation decisions.

## Repository Structure
```
HelmTrack/
├── notebooks/              # Jupyter notebooks (B1-B6)
├── api/                    # FastAPI model serving (B7)
├── app.py                  # Streamlit UI
├── model/                  # Trained model weights 
├── mlflow_screenshots/     # MLflow experiment comparison screenshots
├── .env.example            # Environment variable template
└── requirements.txt        # Python dependencies
```

## Setup

1. Install PyTorch first from https://pytorch.org/get-started/locally/ (choose your CUDA version or CPU-only)

2. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in your Roboflow API key:
```bash
cp .env.example .env
```

3. Download the dataset by running `notebooks/data_exploration.ipynb` (requires Roboflow API key in `.env`)
   - Dataset is hosted at: https://app.roboflow.com/peerapat-seenoi/helmet-detection-nsbwm-luu1f/1
   - You need a Roboflow account and access to the project to download

4. Obtain model weights:
   - `model/best.pt` is not included in the repo due to file size
   - Run `notebooks/model_training.ipynb` to retrain from scratch

## Running the API
```bash
uvicorn api.main:app --reload
```
API docs available at `http://localhost:8000/docs`

## Running the UI
```bash
streamlit run app.py
```
Open `http://localhost:8501` — requires the API to be running first.

## Model
- Architecture: YOLOv8s (small)
- Dataset: 1,078 train / 308 valid / 154 test images
- Classes: Helmet, NoHelmet, Motorbike, PNumber
- Best run: run3_yolov8s_30ep — mAP@50 = 0.8362

## Notebooks
| Notebook | Description |
|---|---|
| data_exploration | Dataset download, class distribution, sample images |
| model_training | 3 YOLOv8 training runs with MLflow tracking, confusion matrix, best model selection |
| fairness_analysis | Per-class precision, recall, mAP fairness gap analysis |
| explainability | Model detections and confidence score distribution on test images |
| per_prediction_reasoning | Human-readable reasoning string for every bounding box prediction |
