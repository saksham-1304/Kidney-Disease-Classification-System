# 🏥 Kidney Disease Classification System
## Deep Learning CNN Model with MLflow & DVC Pipeline

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12.0-FF6F00.svg)](https://tensorflow.org/)
[![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-blue.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue.svg)](https://mlflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://docker.com/)

</div>

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Training Pipeline](#training-pipeline)
- [MLflow Integration](#mlflow-integration)
- [DVC Pipeline](#dvc-pipeline)
- [API Endpoints](#api-endpoints)
- [Model Performance](#model-performance)
- [Docker Deployment](#docker-deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Project Overview

This is a **production-ready Deep Learning application** for classifying kidney diseases from CT scan images using Convolutional Neural Networks (CNN). The project demonstrates a complete ML workflow including data ingestion, model training, evaluation, and deployment with proper experiment tracking and data versioning.

### **Disease Classification Categories:**
- **Cyst**
- **Normal**
- **Stone**
- **Tumor**

---

## 🏥 Problem Statement

Kidney disease is a silent killer affecting millions worldwide. Early detection through medical imaging is crucial for treatment. This project automates the classification of kidney CT scans to:

1. **Accelerate diagnosis** - Reduce manual analysis time
2. **Improve accuracy** - Leverage deep learning capabilities
3. **Scale screening** - Enable large-scale automated analysis
4. **Support radiologists** - Provide decision support (not replacement)

---

## 🏗️ Architecture

The project follows a **modular, production-grade architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│          Flask Web Application (app.py)                 │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Frontend: HTML/Bootstrap UI for Image Upload    │   │
│  │  API Endpoints: /predict, /train, /              │   │
│  └──────────────────────────────────────────────────┘   │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│      CNN Classifier Core (cnnClassifier)                │
├─────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Components   │  │ Config       │  │ Pipeline     │  │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤  │
│  │ • Data       │  │ • YAML       │  │ • Stage 1    │  │
│  │   Ingestion  │  │   Config     │  │   Data       │  │
│  │ • Base       │  │ • Entity     │  │   Ingestion  │  │
│  │   Model      │  │   Definition │  │ • Stage 2    │  │
│  │   Prep       │  │              │  │   Base Model │  │
│  │ • Training   │  │              │  │ • Stage 3    │  │
│  │ • Evaluation │  │              │  │   Training   │  │
│  │ • Prediction │  │              │  │ • Stage 4    │  │
│  │              │  │              │  │   Evaluation │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└──────────────┬──────────────────────────────────────────┘
               │
┌──────────────▼──────────────────────────────────────────┐
│  ML Ops Layer (MLflow + DVC)                           │
├─────────────────────────────────────────────────────────┤
│  • Experiment Tracking (MLflow)                         │
│  • Data Version Control (DVC)                           │
│  • Model Registry & Artifacts                           │
│  • Metrics & Parameters Logging                         │
└─────────────────────────────────────────────────────────┘
```

---

## ✨ Key Features

- ✅ **End-to-End Pipeline**: From data ingestion to model deployment
- ✅ **MLflow Integration**: Experiment tracking, parameter logging, and artifact management
- ✅ **DVC Pipeline**: Data versioning and reproducible ML workflows
- ✅ **Flask Web API**: REST API for predictions and training
- ✅ **Data Augmentation**: Improves model robustness with image transformations
- ✅ **Transfer Learning**: VGG16 base model for faster training and better accuracy
- ✅ **Modular Architecture**: Clean code with separation of concerns
- ✅ **Docker Support**: Containerized deployment ready
- ✅ **Configuration Management**: YAML-based configuration for flexibility
- ✅ **Comprehensive Logging**: Detailed pipeline execution logs

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Deep Learning** | TensorFlow 2.12.0, Keras |
| **Base Architecture** | VGG16 (Transfer Learning) |
| **Data Processing** | Pandas, NumPy, Matplotlib, Seaborn |
| **Web Framework** | Flask, Flask-CORS |
| **ML Ops** | MLflow 2.2.2, DVC |
| **Configuration** | PyYAML, Python-box |
| **Utilities** | Joblib, gdown, ensure, scipy |
| **Containerization** | Docker |
| **Python Version** | 3.8+ |

---

## 📂 Project Structure

```
Kidney-Disease-Classification-System/
│
├── artifacts/                              # Generated outputs
│   ├── data_ingestion/                    # Downloaded & extracted data
│   │   ├── kidney-ct-scan-image/
│   │   │   ├── CT_KIDNEY_DISEASE_CYST/
│   │   │   ├── CT_KIDNEY_DISEASE_NORMAL/
│   │   │   └── CT_KIDNEY_DISEASE_STONE/
│   │   └── data.zip
│   ├── prepare_base_model/                # Base model files
│   │   ├── base_model.h5
│   │   └── base_model_updated.h5
│   ├── training/                          # Trained model
│   │   └── model.h5
│   └── evaluation/                        # Evaluation results
│
├── src/cnnClassifier/                     # Main package
│   ├── __init__.py                        # Logger initialization
│   ├── components/                        # Core logic
│   │   ├── data_ingestion.py             # Downloads & extracts data
│   │   ├── prepare_base_model.py         # VGG16 base model prep
│   │   ├── model_training.py             # Training logic
│   │   ├── model_evaluation_mlflow.py    # Evaluation & MLflow logging
│   │   └── __init__.py
│   │
│   ├── config/                            # Configuration management
│   │   ├── configuration.py               # Config loader
│   │   └── __init__.py
│   │
│   ├── entity/                            # Data classes
│   │   ├── config_entity.py              # Config entities
│   │   └── __init__.py
│   │
│   ├── pipeline/                          # ML pipeline stages
│   │   ├── stage_01_data_ingestion.py    # Data ingestion pipeline
│   │   ├── stage_02_prepare_base_model.py# Base model preparation
│   │   ├── stage_03_model_training.py    # Model training
│   │   ├── stage_04_model_evaluation.py  # Model evaluation
│   │   ├── prediction.py                 # Prediction pipeline
│   │   └── __init__.py
│   │
│   ├── utils/                             # Utility functions
│   │   ├── common.py                     # Common utilities
│   │   └── __init__.py
│   │
│   └── constants/                         # Constants
│       └── __init__.py
│
├── config/                                # Configuration files
│   └── config.yaml                       # Main configuration
│
├── research/                              # Jupyter notebooks
│   ├── 01_data_ingestion.ipynb
│   ├── 02_prepare_base_model.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_model_evaluation_with_mlflow.ipynb
│   └── trials.ipynb
│
├── templates/                             # Web frontend
│   └── index.html                        # Main UI for predictions
│
├── model/                                 # Saved models
│   └── model.h5                          # Final trained model
│
├── report/                                # Project documentation
│   ├── main.tex
│   ├── project_report.tex
│   ├── refs.bib
│   ├── IEEEbib.bst
│   └── spconf.sty
│
├── app.py                                 # Flask application
├── main.py                                # Training pipeline orchestrator
├── setup.py                               # Package setup
├── requirements.txt                       # Dependencies
├── params.yaml                            # Model parameters
├── dvc.yaml                               # DVC pipeline definition
├── Dockerfile                             # Docker configuration
├── LICENSE                                # MIT License
└── README.md                              # This file
```

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8 or higher
- Anaconda/Miniconda (recommended)
- 4GB+ RAM for training
- GPU (optional, but recommended for faster training)

### **STEP 1: Clone the Repository**

```bash
git clone https://github.com/saksham-1304/Kidney-Disease-Classification-System.git
cd Kidney-Disease-Classification-System
```

### **Colab Quick Start (Recommended for GPU training)**

Use the notebook [colab_training.ipynb](colab_training.ipynb) to run the pipeline on Google Colab GPU.

```bash
# In Colab (after cloning repo)
python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
python src/cnnClassifier/pipeline/stage_03_model_training.py
python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
```

### **STEP 2: Create Conda Environment**

```bash
# Create virtual environment
conda create -n cnncls python=3.8 -y

# Activate environment
conda activate cnncls
```

### **STEP 3: Install Dependencies**

```bash
# Install required packages
pip install -r requirements.txt

# Install the package in editable mode
pip install -e .
```

### **STEP 4: Configure MLflow (Optional)**

```bash
# Set MLflow tracking URI (uses local SQLite by default)
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Optional: Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
# Access at http://localhost:5000
```

---

## 💻 Usage

### **1. Run the Complete Training Pipeline**

Executes all 4 stages: data ingestion → base model prep → training → evaluation

```bash
python main.py
```

**Or using DVC (for reproducible runs):**

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro stage_name
```

### **2. Start the Flask Web Application**

```bash
python app.py
```

**Access the web interface:**
- Open browser: `http://localhost:8080`
- Upload a kidney CT scan image
- Get instant predictions with confidence score

### **3. Run Prediction on a Single Image**

```python
from cnnClassifier.pipeline.prediction import PredictionPipeline

# Initialize predictor
predictor = PredictionPipeline("path/to/image.jpg")
result = predictor.predict()
print(result)
```

### **4. View MLflow Experiments**

```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

Navigate to `http://localhost:5000` to view:
- Experiment runs
- Metrics (loss, accuracy)
- Logged parameters
- Model artifacts

---

## 🔄 Training Pipeline

The training follows a 4-stage pipeline defined in `dvc.yaml`:

### **Stage 1: Data Ingestion**
- **File**: `stage_01_data_ingestion.py`
- **Task**: Download CT scan dataset from Google Drive
- **Output**: Extracted images organized by class

```bash
dvc repro data_ingestion
```

### **Stage 2: Prepare Base Model**
- **File**: `stage_02_prepare_base_model.py`
- **Task**: Load VGG16 from ImageNet, freeze base layers, add custom top
- **Parameters**: IMAGE_SIZE, INCLUDE_TOP, CLASSES, WEIGHTS, LEARNING_RATE
- **Output**: `base_model_updated.h5`

```bash
dvc repro prepare_base_model
```

### **Stage 3: Model Training**
- **File**: `stage_03_model_training.py`
- **Task**: Train the model with data augmentation
- **Parameters**: IMAGE_SIZE, EPOCHS, BATCH_SIZE, AUGMENTATION
- **Output**: `model.h5`

```bash
dvc repro training
```

### **Stage 4: Model Evaluation**
- **File**: `stage_04_model_evaluation.py`
- **Task**: Evaluate on test set, log metrics to MLflow
- **Output**: Accuracy, loss, confusion matrix

```bash
dvc repro evaluation
```

---

## 📊 Model Architecture

The model uses **Transfer Learning** with VGG16:

```
Input Image (224 × 224 × 3)
         ↓
   VGG16 Base (Pre-trained on ImageNet)
   - 16 Convolutional layers
   - 3 MaxPooling layers
   - Frozen weights
         ↓
   Flatten Layer
         ↓
   Dense Layer (256 units, ReLU)
         ↓
   Dropout (50%)
         ↓
   Output Layer (2 units, Softmax)
         ↓
   Predictions (Diseased vs Normal)
```

### **Key Architecture Decisions:**
- **Base Model**: VGG16 (pretrained on ImageNet)
- **Top Layers**: Custom fully connected layers
- **Dropout**: 50% to prevent overfitting
- **Activation**: ReLU for hidden, Softmax for output
- **Loss**: Binary Crossentropy
- **Optimizer**: Adam

---

## 📈 Model Performance

Current model metrics (`scores.json`):

```json
{
    "loss": 24.86,
    "accuracy": 0.52
}
```

**Note**: Further training and parameter tuning can improve accuracy. Current results may reflect early testing stage.

---

## 🔌 API Endpoints

### **1. Home Page**
```
GET /
Returns: HTML interface for image upload and prediction
```

### **2. Make Prediction**
```
POST /predict
Request Body:
{
    "image": "<base64_encoded_image>"
}

Response:
{
    "class": "Normal" | "Diseased",
    "probability": 0.95
}
```

### **3. Train Model**
```
GET /train
Action: Triggers training pipeline (python main.py)
Response: "Training done successfully!"
Note: Training runs asynchronously
```

---

## 📦 DVC Pipeline

DVC ensures reproducible ML workflows with automatic dependency tracking:

```yaml
stages:
  data_ingestion:
    cmd: python src/cnnClassifier/pipeline/stage_01_data_ingestion.py
    deps:
      - src/cnnClassifier/pipeline/stage_01_data_ingestion.py
      - config/config.yaml
    outs:
      - artifacts/data_ingestion/kidney-ct-scan-image

  prepare_base_model:
    cmd: python src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
    deps:
      - src/cnnClassifier/pipeline/stage_02_prepare_base_model.py
      - config/config.yaml
    params:
      - IMAGE_SIZE
      - INCLUDE_TOP
      - CLASSES
      - WEIGHTS
      - LEARNING_RATE
    outs:
      - artifacts/prepare_base_model

  training:
    cmd: python src/cnnClassifier/pipeline/stage_03_model_training.py
    deps:
      - src/cnnClassifier/pipeline/stage_03_model_training.py
      - config/config.yaml
      - artifacts/data_ingestion/kidney-ct-scan-image
      - artifacts/prepare_base_model
    params:
      - IMAGE_SIZE
      - EPOCHS
      - BATCH_SIZE
      - AUGMENTATION
    outs:
      - artifacts/training/model.h5

  evaluation:
    cmd: python src/cnnClassifier/pipeline/stage_04_model_evaluation.py
    deps:
      - src/cnnClassifier/pipeline/stage_04_model_evaluation.py
      - config/config.yaml
      - artifacts/data_ingestion/kidney-ct-scan-image
      - artifacts/training/model.h5
    params:
      - IMAGE_SIZE
```

**DVC Commands:**

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro training

# View pipeline DAG
dvc dag

# Check pipeline status
dvc status
```

---

## 🧪 MLflow Integration

MLflow tracks experiments, logs metrics, and manages artifacts:

### **Logged Information:**
- **Parameters**: IMAGE_SIZE, EPOCHS, BATCH_SIZE, LEARNING_RATE
- **Metrics**: Training loss, validation accuracy, test accuracy
- **Artifacts**: Trained model, predictions

### **Start MLflow UI:**

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access at: `http://localhost:5000`

### **View in Code:**

```python
import mlflow

with mlflow.start_run():
    mlflow.log_param("EPOCHS", 100)
    mlflow.log_param("BATCH_SIZE", 16)
    
    # ... Training code ...
    
    mlflow.log_metric("accuracy", 0.92)
    mlflow.log_metric("loss", 0.24)
    mlflow.log_artifact("model.h5")
```

---

## 🐳 Docker Deployment

### **Build Docker Image**

```bash
docker build -t kidney-disease-classifier:latest .
```

### **Run Container**

```bash
# Run Flask app
docker run -p 8080:8080 kidney-disease-classifier:latest

# With volume mount for persistence
docker run -p 8080:8080 \
    -v $(pwd)/artifacts:/app/artifacts \
    kidney-disease-classifier:latest
```

### **Docker Compose (Optional)**

```bash
docker-compose up -d
```

---

## 📝 Configuration

### **params.yaml** - Model Hyperparameters

```yaml
AUGMENTATION: True           # Enable/disable data augmentation
IMAGE_SIZE: [224, 224, 3]   # Input image size (VGG16 requirement)
BATCH_SIZE: 16              # Training batch size
INCLUDE_TOP: False          # Use VGG16 without top layers
EPOCHS: 1                   # Number of training epochs
CLASSES: 2                  # Number of output classes
WEIGHTS: imagenet           # Pretrained weights
LEARNING_RATE: 0.01         # Optimizer learning rate
```

### **config.yaml** - File Paths & URLs

```yaml
artifacts_root: artifacts

data_ingestion:
  source_URL: https://drive.google.com/file/d/...
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  trained_model_path: artifacts/training/model.h5
```

---

## 🔧 Workflows & Development

### **Standard Development Workflow:**

1. **Update config.yaml** - Modify paths and configurations
2. **Update params.yaml** - Adjust model hyperparameters
3. **Update entity** - Define configuration data classes
4. **Update configuration manager** - Load and manage configs
5. **Update components** - Implement logic for each stage
6. **Update pipeline** - Orchestrate pipeline stages
7. **Update main.py** - Execute training pipeline
8. **Update dvc.yaml** - Define DVC stages
9. **Update app.py** - Add API endpoints as needed

---

## 🧑‍💻 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact & Support

- **Original Author**: [krishnaik06](https://github.com/krishnaik06)
- **Repository**: [Kidney-Disease-Classification-Deep-Learning-Project](https://github.com/krishnaik06/Kidney-Disease-Classification-Deep-Learning-Project)

---

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- MLflow for experiment tracking
- DVC for data version control
- VGG16 authors for the powerful base model architecture
- Dataset providers

---

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [MLflow Documentation](https://mlflow.org/)
- [DVC Documentation](https://dvc.org/)
- [VGG16 Paper](https://arxiv.org/abs/1409.1556)
- [Transfer Learning Guide](https://en.wikipedia.org/wiki/Transfer_learning)

---

<div align="center">

**Made with ❤️ for Medical AI**

If you found this project helpful, please give it a ⭐ on GitHub!

</div>






## MLflow

- [Documentation](https://mlflow.org/docs/latest/index.html)

- [MLflow tutorial](https://youtu.be/qdcHHrsXA48?si=bD5vDS60akNphkem)

##### cmd
- mlflow ui

### dagshub
[dagshub](https://dagshub.com/)

MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow \
MLFLOW_TRACKING_USERNAME=entbappy \
MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0 \
python script.py

Run this to export as env variables:

```bash

export MLFLOW_TRACKING_URI=https://dagshub.com/entbappy/Kidney-Disease-Classification-MLflow-DVC.mlflow

export MLFLOW_TRACKING_USERNAME=entbappy 

export MLFLOW_TRACKING_PASSWORD=6824692c47a369aa6f9eac5b10041d5c8edbcef0

```


### DVC cmd

1. dvc init
2. dvc repro
3. dvc dag


## About MLflow & DVC

MLflow

 - Its Production Grade
 - Trace all of your expriements
 - Logging & taging your model


DVC 

 - Its very lite weight for POC only
 - lite weight expriements tracker
 - It can perform Orchestration (Creating Pipelines)



# AWS-CICD-Deployment-with-Github-Actions

## 1. Login to AWS console.

## 2. Create IAM user for deployment

	#with specific access

	1. EC2 access : It is virtual machine

	2. ECR: Elastic Container registry to save your docker image in aws


	#Description: About the deployment

	1. Build docker image of the source code

	2. Push your docker image to ECR

	3. Launch Your EC2 

	4. Pull Your image from ECR in EC2

	5. Lauch your docker image in EC2

	#Policy:

	1. AmazonEC2ContainerRegistryFullAccess

	2. AmazonEC2FullAccess

	
## 3. Create ECR repo to store/save docker image
    - Save the URI: 566373416292.dkr.ecr.us-east-1.amazonaws.com/chicken

	
## 4. Create EC2 machine (Ubuntu) 

## 5. Open EC2 and Install docker in EC2 Machine:
	
	
	#optinal

	sudo apt-get update -y

	sudo apt-get upgrade
	
	#required

	curl -fsSL https://get.docker.com -o get-docker.sh

	sudo sh get-docker.sh

	sudo usermod -aG docker ubuntu

	newgrp docker
	
# 6. Configure EC2 as self-hosted runner:
    setting>actions>runner>new self hosted runner> choose os> then run command one by one


# 7. Setup github secrets:

    AWS_ACCESS_KEY_ID=

    AWS_SECRET_ACCESS_KEY=

    AWS_REGION = us-east-1

    AWS_ECR_LOGIN_URI = demo>>  566373416292.dkr.ecr.ap-south-1.amazonaws.com

    ECR_REPOSITORY_NAME = simple-app


