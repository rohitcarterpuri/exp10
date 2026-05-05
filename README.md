# 🖼️ Image Classification using Transfer Learning (VGG16)

This project demonstrates transfer learning using a pre-trained VGG16 model for image classification.

## 📌 Features
- Uses pre-trained VGG16 (ImageNet weights)
- Custom classifier on top
- Image data generator for preprocessing
- Training & evaluation pipeline

## 🛠 Tech Stack
- Python
- TensorFlow / Keras
- NumPy

## 📂 Project Structure
```
vgg16_transfer_learning_project/
│
├── data/
│   ├── train/
│   └── val/
├── src/
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Organize dataset
```
data/train/class1/
data/train/class2/
data/val/class1/
data/val/class2/
```

### 3. Train model
```bash
python src/train.py
```

## 🧠 Model
- VGG16 base (frozen)
- Custom dense layers

## 📜 License
MIT
