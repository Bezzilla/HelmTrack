# HelmTrack

AI-powered motorcycle helmet violation detection system using YOLOv8.

## Team
- Peerapat Seenoi
- Tanapoom Srikaewkheaw

## Project Overview
HelmTrack detects motorcycle riders without helmets from CCTV footage using YOLOv8 object detection, and logs violation records with license plate information via OCR.

## Repository Structure
```
HelmTrack/
├── notebooks/          # Jupyter notebooks for POC (Parts B1-B6)
├── api/                # FastAPI model serving (Part B7)
├── model/              # Trained model weights
├── data/sample/        # Sample images for testing
└── mlflow_screenshots/ # MLflow experiment comparison screenshots
```

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Downloaded from Roboflow Universe — helmet detection dataset in YOLOv8 format.
Place dataset in `data/` after downloading.

## Running the API
```bash
uvicorn api.main:app --reload
```
Then go to `http://localhost:8000/docs` to test the API.

## Notebooks
| Notebook | Description |
|---|---|
| B1_data_exploration | Dataset overview, class distribution, sample images |
| B2_model_training | YOLOv8 fine-tuning, confusion matrix, metrics + MLflow tracking (3 runs) |
| B3_fairness_analysis | Per-class fairness metrics |
| B5_explainability | GradCAM feature visualization |
| B6_prediction_reasoning | Per-prediction explanation with GradCAM |

