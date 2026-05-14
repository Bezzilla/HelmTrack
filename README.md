# HelmTrack

AI-powered motorcycle helmet violation detection system using YOLOv8.

## Team
- **Tanapoom Srikaewkheaw** : System design, UI/UX design, API interface design
- **Peerapat Seenoi** : Data collection, model training, MLflow experiment tracking, model evaluation, API deployment

## How It Works
1. User uploads a traffic image
2. The image is sent to the FastAPI backend for inference
3. YOLOv8 runs object detection, identifying Helmet, No Helmet, Motorbike, and License Plate
4. If any NoHelmet is detected, the system flags a violation
5. Results are displayed with bounding boxes, confidence scores, and a verdict

## Project Overview
HelmTrack detects motorcycle riders without helmets from traffic images using YOLOv8 object detection. The system identifies 4 classes: Helmet, NoHelmet, Motorbike, and PNumber (license plate). A FastAPI backend serves the model and a Streamlit UI provides a proof-of-concept interface for uploading images and receiving instant violation decisions.

## Grading Guide

| Task | File / Location |
|---|---|
| B1: Data Exploration | `notebooks/data_exploration.ipynb` |
| B2: Model Training | `notebooks/model_training.ipynb` |
| B3: Fairness Analysis | `notebooks/fairness_analysis.ipynb` |
| B4: MLflow Screenshots | `mlflow_screenshots/` |
| B5: Model Explainability | `notebooks/explainability.ipynb` |
| B6: Prediction Reasoning | `notebooks/per_prediction_reasoning.ipynb` |
| B7: Model Deployment | `notebooks/api_documentation.ipynb` + `api/main.py` |
| Model Artifact | `model/best.pt` |
| UI (Part C) | `app.py` (Streamlit) |
| API (Part C) | `api/main.py` (FastAPI) |

## Repository Structure
```
HelmTrack/
├── notebooks/              # Jupyter notebooks (B1-B7)
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   ├── fairness_analysis.ipynb
│   ├── explainability.ipynb
│   ├── per_prediction_reasoning.ipynb
│   └── api_documentation.ipynb
├── api/                    # FastAPI model serving
│   └── main.py
├── app.py                  # Streamlit UI
├── model/                  # Trained model weights
│   └── best.pt
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

3. Copy `.env.example` to `.env` and fill in your Roboflow API key:
```bash
cp .env.example .env
```

4. Download the dataset by running `notebooks/data_exploration.ipynb` (requires Roboflow API key in `.env`)
   - Dataset is hosted at: https://app.roboflow.com/peerapat-seenoi/helmet-detection-nsbwm-luu1f/1

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
- Model artifact: `model/best.pt`

## Notebooks

| Notebook | Task | Description |
|---|---|---|
| data_exploration | B1 | Dataset download, class distribution, sample images |
| model_training | B2 | 3 YOLOv8 training runs with MLflow tracking, confusion matrix, best model selection |
| fairness_analysis | B3 | Per-class precision, recall, mAP fairness gap analysis |
| explainability | B5 | Model detections and confidence score distribution on test images |
| per_prediction_reasoning | B6 | Human-readable reasoning string for every bounding box prediction |
| api_documentation | B7 | API contract, 3 test scenarios, scalability considerations |
