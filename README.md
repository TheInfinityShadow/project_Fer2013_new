# FER2013 Emotion Recognition

This project is a deep learning-based emotion recognition system built on the FER2013 dataset. It uses a fine-tuned ResNet50V2 backbone with CBAM attention and Focal Loss to improve recognition performance, especially for underrepresented classes. The training pipeline is modular, reproducible, and includes tools for analysis and interpretability.

---

## 📁 Project Structure

```
FER2013-Emotion-Recognition/
├── configs/                   # Configuration files
│   ├── default_config.py      
│   └── custom_config.py  
├── data/                      # FER2013 dataset
│   ├── train/                 
│   ├── test/
├── docs/                      # Documentation
│   ├── setup.md              	# Installation and setup instructions
│   └── api.md                	# (Optional) API documentation
├── models/                    # Trained model files
│   ├── checkpoints/          	# Model checkpoints (.h5, .keras)
│   └── final/                	# Final trained models
├── notebooks/                 # Jupyter notebooks
│   └── experiments.ipynb    	# (Optional) For prototyping or visualizing results
├── results/                   # (Renamed from outputs/) Experiment outputs
│   ├── logs/                 	# Training logs
│   ├── plots/                	# Accuracy/loss plots, confusion matrix
│   └── metrics/              	# (New) Evaluation metrics (e.g., CSV, JSON)
├── src/                       # Source code
│   ├── data/                 	# Data-related scripts
│   │   ├── __init__.py
│   │   ├── data_loader.py    # Loads and preprocesses dataset
│   │   └── augment.py        # Image augmentation (e.g., ImageDataGenerator)
│   ├── models/              	# Model-related scripts
│   │   ├── __init__.py
│   │   ├── build_model.py    # ResNet50V2 + CBAM model
│   │   └── cbam.py           # CBAM attention module
│   ├── training/            	# Training-related scripts
│   │   ├── __init__.py
│   │   ├── callbacks.py 	# ModelCheckpoint and EarlyStopping
│   │   ├── learning_rate.py 	# CosineDecay
│   │   ├── training.py      	# Model training and saving
│   │   ├── loss.py          	# Focal loss implementation
│   │   └── lr_scheduler.py  	# LogCosineDecay custom callback
│   └── utils/               	# Utility scripts
│       ├── __init__.py
│       ├── evaluation.py    	# Evaluation and confusion matrix
│       └── plotting.py      	# Training loss/accuracy plots
│   
├── tests/                    	# (New) Unit tests
│   ├── test_data.py          	# Tests for data loading
│   └── test_model.py         	# Tests for model
├── main.py                    # Main script to run training + evaluation
├── requirements.txt           # Dependencies
├── environment.yml            # (New, optional) Conda environment file
├── README.md                  # Project overview, setup, usage
├── LICENSE                    # (New) License file
└── .gitignore                 # Ignored files (ensure large files excluded)

```

---

## 🚀 Features

* ✅ CBAM attention module for spatial and channel refinement
* ✅ Focal Loss to handle class imbalance
* ✅ Mixed precision-ready structure
* ✅ Learning rate scheduler (Cosine Decay)
* ✅ Confusion matrix + classification report
* ✅ Reproducible code and modular design

---

## 📆 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

**Dependencies:**

* TensorFlow
* OpenCV
* scikit-learn
* matplotlib
* seaborn

---

## 🧠 Model Architecture

* **Base model**: ResNet50V2 (last 50 layers unfrozen)
* **Attention**: CBAM (Convolutional Block Attention Module)
* **Classifier**: Dense → Dropout → BatchNorm → Dense → Softmax
* **Loss**: Focal Loss
* **Optimizer**: RMSprop with cosine decay schedule

---

## 🗂️ Dataset

FER2013 is a dataset of grayscale 48x48 pixel facial expression images divided into 7 emotion categories:

* Angry
* Disgust
* Fear
* Happy
* Sad
* Surprise
* Neutral

📅 **Note**: The dataset is not included in this repository. Download FER2013 from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) and place images under `data/train/`, organized into folders per class.

---

## 🔧 How to Use

### 1. Prepare Data

Organize the dataset like this:

```
data/train/
  angry/
  happy/
  sad/
  ...
```

### 2. Train and Evaluate

```bash
python main.py
```

* Trains the model with augmentations and focal loss
* Saves the best model and training plots
* Prints classification report and saves confusion matrix

---

## 📊 Outputs

Saved in the `outputs/` folder:

* `checkpoints/final_model.h5`: Trained model weights
* `plots/accuracy.png`: Accuracy over epochs
* `plots/loss.png`: Loss over epochs
* `plots/confusion_matrix.png`: Confusion matrix

---

## 📊 Test Results (test Set)

| Emotion     | Precision | Recall | F1 Score | support |
|-------------|-----------|--------|----------|---------|
| Angry       | 0.56      | 0.55   | 0.56     | 958     |
| Disgust     | 0.68      | 0.52   | 0.59     | 111     |
| Fear        | 0.56      | 0.33   | 0.42     | 1024    |
| Happy       | 0.85      | 0.87   | 0.86     | 1774    |
| Neutral     | 0.57      | 0.66   | 0.62     | 1233    |
| Sad	      | 0.52      | 0.56   | 0.54     | 1247    |
| Surprise    | 0.70      | 0.81   | 0.75     | 831     |
| Accuracy    |      	  |    	   | 0.65     | 7178	|

*Note: Results will vary based on training sample size and data quality.*

---

## 🔍 Inference on FER2013 Test Set

To evaluate the model on the official FER2013 test set:

```bash
python src/test.py
```

## 📜 License

This project is released for educational and research purposes. Please cite the original FER2013 dataset if used in publications.
