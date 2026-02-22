# Kidney Disease Classification System

**Deep Learning Pipeline with VGG16, K-Fold Cross-Validation, DVC, and MLflow**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%E2%89%A5%202.16-FF6F00.svg)](https://tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Pipeline-blue.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue.svg)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

A **production-grade deep learning system** for classifying kidney CT scan images into four categories:

| Class | Description |
|-------|-------------|
| **Normal** | Healthy kidney |
| **Cyst** | Fluid-filled sac |
| **Stone** | Calcium/mineral deposit |
| **Tumor** | Abnormal tissue growth |

The project implements a complete ML lifecycle: automated data ingestion from Kaggle, stratified 5-fold cross-validation, model training with augmentation, per-fold evaluation with aggregated metrics, and a Flask web app for inference.

---

## Architecture

```
Flask Web App (app.py)
    |
    v
Prediction Pipeline  <--  model/model.h5 (final trained model)
    |
    v
cnnClassifier Package (src/cnnClassifier/)
    |-- components/      Data ingestion, model prep, training, evaluation
    |-- config/          ConfigurationManager (reads YAML -> frozen dataclasses)
    |-- entity/          Frozen @dataclass definitions for each stage
    |-- pipeline/        Per-stage orchestration scripts + prediction
    |-- utils/           YAML, JSON, base64 helpers
    |
    v
MLOps Layer
    |-- DVC              Pipeline DAG, dependency tracking, reproducibility
    |-- MLflow           Hyperparameter logging, metric tracking, model registry
```

### Model Architecture

```
Input (224 x 224 x 3)
        |
  VGG16 Base (ImageNet weights, all layers frozen)
  [13 conv layers + 5 max-pool layers]
  14,714,688 non-trainable parameters
        |
  Flatten (7 x 7 x 512 -> 25,088)
        |
  Dropout (p=0.5)
        |
  Dense (4 units, softmax)  -- 100,356 trainable parameters
        |
  Output: [Cyst, Normal, Stone, Tumor]
```

**Optimizer:** Adam (LR=0.001, with ReduceLROnPlateau)
**Loss:** Categorical Cross-Entropy
**Callbacks:** EarlyStopping (patience=5), ReduceLROnPlateau (patience=3, factor=0.5)

---

## Model Performance

5-fold stratified cross-validation on **12,446 CT images** (VGG16 + Dropout(0.5) head, Adam, augmentation enabled).

### Aggregate Results

| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | **96.12%** | ±0.82% |
| Loss | 0.1231 | ±0.0244 |
| Macro Precision | 95.77% | ±0.88% |
| Macro Recall | 94.77% | ±1.15% |
| Macro F1 | **95.23%** | ±1.04% |

### Per-Fold Breakdown

| Fold | Accuracy | Loss | Macro F1 | Val Samples |
|------|----------|------|----------|-------------|
| 1 | 94.58% | 0.1713 | 93.28% | 2,491 |
| 2 | 96.23% | 0.1116 | 95.39% | 2,491 |
| 3 | **97.03%** | 0.1103 | **96.39%** | 2,489 |
| 4 | 96.42% | 0.1164 | 95.66% | 2,488 |
| 5 | 96.34% | 0.1057 | 95.42% | 2,487 |
| **Mean** | **96.12%** | **0.1231** | **95.23%** | 12,446 total |

### Per-Class Metrics (averaged across 5 folds)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Cyst | 95.88% | 97.71% | 96.77% |
| Normal | 97.05% | 97.73% | 97.39% |
| Stone | 94.91% | 89.98% | 92.36% |
| Tumor | 95.23% | 93.65% | 94.38% |

> Stone has the lowest recall (89.98%) due to its smaller support (~275 val images per fold). Normal achieves the highest F1 (97.39%).

---

## Project Structure

```
Kidney-Disease-Classification-System/
|
|-- app.py                      Flask web application
|-- main.py                     Training pipeline orchestrator
|-- Dockerfile                  Docker container config
|-- dvc.yaml                    DVC pipeline definition (4 stages)
|-- params.yaml                 Hyperparameters (epochs, LR, k-folds, etc.)
|-- setup.py                    Package setup
|-- requirements.txt            Python dependencies
|-- colab_training.ipynb        Google Colab training notebook
|-- kaggle_training.ipynb       Kaggle training notebook
|
|-- config/
|   |-- config.yaml             Paths, URLs, MLflow URI
|
|-- src/cnnClassifier/
|   |-- __init__.py             Logger setup
|   |-- components/
|   |   |-- data_ingestion.py           Kaggle download + k-fold splitting
|   |   |-- prepare_base_model.py       VGG16 + custom head
|   |   |-- model_training.py           Per-fold + final model training
|   |   |-- model_evaluation_mlflow.py  Per-fold evaluation + MLflow
|   |-- config/
|   |   |-- configuration.py           ConfigurationManager
|   |-- entity/
|   |   |-- config_entity.py           Frozen dataclasses
|   |-- pipeline/
|   |   |-- stage_01_data_ingestion.py
|   |   |-- stage_02_prepare_base_model.py
|   |   |-- stage_03_model_training.py
|   |   |-- stage_04_model_evaluation.py
|   |   |-- prediction.py              Inference pipeline
|   |-- constants/
|   |   |-- __init__.py                File path constants
|   |-- utils/
|       |-- common.py                  YAML/JSON/base64 utilities
|
|-- model/
|   |-- model.h5                Final trained model (for serving)
|
|-- artifacts/                  Generated during pipeline (git-ignored)
|   |-- data_ingestion/
|   |   |-- kidney-ct-scan-dataset/    Raw download
|   |   |-- folds/fold_{1..5}/         K-fold train/val splits
|   |   |-- all/                       All images (for final model)
|   |-- prepare_base_model/
|   |   |-- base_model.h5
|   |   |-- base_model_updated.h5
|   |-- training/
|       |-- model.h5                   Final model
|       |-- model_fold_{1..5}.h5       Per-fold models
|
|-- report/                     IEEE-format project report + LaTeX source
|-- research/                   Development Jupyter notebooks
|-- templates/
    |-- index.html              Web interface
```

---

## Quick Start

### Option A: Train on Kaggle (Recommended)

1. Push this repo to GitHub
2. Create a new Kaggle Notebook, enable **GPU** and **Internet**
3. Upload `kaggle_training.ipynb` and run all cells
4. Download the `trained_outputs.zip` from the output
5. Extract `model/model.h5` into your local `model/` directory

### Option B: Train on Google Colab

1. Open `colab_training.ipynb` in Colab
2. Enable GPU runtime (Runtime > Change runtime type > T4 GPU)
3. Upload your `kaggle.json` when prompted
4. Run all cells; download `trained_outputs.zip` at the end

### Option C: Train Locally

```bash
# Clone
git clone https://github.com/saksham-1304/Kidney-Disease-Classification-System.git
cd Kidney-Disease-Classification-System

# Virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install
pip install -r requirements.txt

# Ensure Kaggle API is configured
# Place kaggle.json in ~/.kaggle/ (Linux/Mac) or %USERPROFILE%\.kaggle\ (Windows)

# Run full pipeline
python main.py
```

### Local Deployment (after training)

```bash
# Ensure model/model.h5 exists (from training or downloaded)
python app.py
# Open http://localhost:8080
```

---

## Pipeline Stages

The pipeline is defined in `dvc.yaml` and can be run with `dvc repro` or `python main.py`.

### Stage 1: Data Ingestion

- Downloads the CT Kidney Dataset from Kaggle (`nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone`, ~12,446 images)
- Creates stratified k-fold splits (k=5, seed=42)
- Copies all images to `all/` directory for final model training

### Stage 2: Prepare Base Model

- Loads VGG16 with ImageNet weights (`include_top=False`)
- Freezes all convolutional layers individually (`layer.trainable = False`)
- Adds: Flatten -> Dropout(0.5) -> Dense(4, softmax)
- Compiles with Adam (LR=0.001) and categorical cross-entropy

### Stage 3: Model Training

- **K-Fold Training:** For each of 5 folds, loads a fresh copy of the base model, trains with augmentation, EarlyStopping, ReduceLROnPlateau, and class weighting
- **Final Model:** Trains on all data (80/20 internal split for callbacks) -> `model.h5`
- Augmentation: rotation (40 deg), horizontal flip, width/height shift, shear, zoom (all 0.2)

### Stage 4: Evaluation

- Evaluates each fold model on its held-out validation set
- Computes per-class precision, recall, F1 and macro averages
- Aggregates metrics across folds (mean +/- std)
- Saves detailed results to `scores.json`

---

## Configuration

### params.yaml

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 50
CLASSES: 4
WEIGHTS: imagenet
LEARNING_RATE: 0.001
K_FOLDS: 5
```

### config/config.yaml

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  source_URL: nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5

evaluation:
  root_dir: artifacts/evaluation
  path_of_model: artifacts/training/model.h5
  data_root: artifacts/data_ingestion
  mlflow_uri: https://dagshub.com/saksham-1304/Kidney-Disease-Classification-System.mlflow
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface for image upload and prediction |
| `/train` | GET/POST | Triggers `python main.py` (runs all 4 stages) |
| `/predict` | POST | Accepts `{"image": "<base64>"}`, returns `[{"image": "Cyst\|Normal\|Stone\|Tumor"}]` |

---

## DVC Pipeline

```bash
# Run full pipeline (only re-runs stages with changed inputs)
dvc repro

# View pipeline DAG
dvc dag

# Check which stages need re-running
dvc status
```

Changing `params.yaml` automatically triggers only the affected downstream stages. For example, changing `EPOCHS` re-runs only training and evaluation; changing `K_FOLDS` re-runs everything from data ingestion.

---

## MLflow

MLflow tracking is configured in `config/config.yaml`. To enable:

1. Set up a DagsHub repo or local MLflow server
2. Update `mlflow_uri` in `config/config.yaml`
3. Uncomment `evaluation.log_into_mlflow()` in `stage_04_model_evaluation.py`

```bash
# Local MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Visit http://localhost:5000
```

---

## Docker

```bash
# Build
docker build -t kidney-classifier .

# Run (ensure model/model.h5 exists)
docker run -p 8080:8080 -v $(pwd)/model:/app/model kidney-classifier

# Access at http://localhost:8080
```

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Deep Learning | TensorFlow ≥2.16 / Keras 3, VGG16 (Transfer Learning) |
| Data | Pandas, NumPy, Matplotlib, Seaborn |
| Web | Flask, Flask-CORS, Bootstrap 4, jQuery |
| MLOps | MLflow, DVC |
| Config | PyYAML, python-box (ConfigBox) |
| Data Source | Kaggle CLI |
| Containerisation | Docker (Python 3.10) |

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Acknowledgments

- [VGG16 Paper](https://arxiv.org/abs/1409.1556) by Simonyan & Zisserman
- [CT Kidney Dataset](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) on Kaggle
- TensorFlow, MLflow, DVC open-source communities
