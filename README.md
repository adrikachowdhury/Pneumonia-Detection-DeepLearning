# Pneumonia Detection using Deep Learning

This project, developed for the **_Machine/Deep Learning Course_**, focuses on detecting pneumonia from chest X-ray images using deep learning techniques. The model is built using transfer learning with the VGG16 architecture as the base model, fine-tuned for the specific task of pneumonia detection. The dataset used is the "Chest X-ray Pneumonia" dataset from Kaggle, which contains labeled images of _normal_ and _pneumonia-infected_ chest X-rays.

## 🚀 Features
- **Data Augmentation**: Techniques like rotation, shifting, shearing, zooming, and flipping are applied to enhance the training data.
- **Model Architecture:** Uses VGG16 as the base model with custom classification layers for binary classification.
- **Training:** The model is trained with data generators to handle large datasets efficiently.
- **Evaluation:** Metrics, primarily macro-average F1-Score, was used to evaluate model performance.

## 📊 Dataset
- **Chest X-Ray Images (Pneumonia)** [Paul Mooney – Kaggle](https://aclanthology.org/S18-1005.pdf)

## 🧠 Model Architecture

The model leverages a pretrained VGG16 architecture followed by a custom classification head for binary classification (Pneumonia vs. Normal):

- **Base Model**: VGG16 (with `include_top=False`, pretrained on ImageNet)
  - Input: `(224, 224, 3)` chest X-ray images
  - Frozen layers: all except the last 15 layers for fine-tuning
- **Custom Head**:
  - `GlobalAveragePooling2D()`
  - `Dense(128, activation='relu')`
  - `Dense(1, activation='sigmoid')` (binary output)

Trained with `binary_crossentropy` loss and evaluated on accuracy, loss, and precision metrics.

## Tools
- Python 3.x, TensorFlow, Keras, OpenCV, NumPy, Matplotlib, scikit-learn, OpenDatasets (for downloading the dataset from Kaggle)

## Setup
1. Download the repository from GitHub.
2. Install dependencies available on requirements.txt
3. Open PneumoniaDetection.ipynb in a notebook (Google Colab/Jupyter) and run all cells to train and evaluate the model through hyperparameter tuning.
