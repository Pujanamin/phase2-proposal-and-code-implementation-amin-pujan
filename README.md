# Automated Fruit Quality and Degradation Assessment Using Deep CNNs and Transfer Learning

An automated agricultural computer vision pipeline developed to classify fruit type and freshness state across 6 distinct categories. This project builds, optimizes, and evaluates a custom VGG-style Deep Convolutional Neural Network (CNN) against an ImageNet-pre-trained MobileNetV2 architecture. 

The models are exposed through an interactive Streamlit web dashboard for real-time model selection and inference tracking.

---

## 📌 Project Overview
* **Domain:** Agricultural Automation, Computer Vision, Deep Learning
* **Dataset:** 3 Popular Fruits with Two Categories: Fresh and Rotten
* **Target Classes (6):** `Fresh Apples`, `Fresh Bananas`, `Fresh Oranges`, `Rotten Apples`, `Rotten Bananas`, `Rotten Oranges`
* **Infrastructure:** Trained using TensorFlow/Keras on a Dual Tesla T4 GPU cluster inside Kaggle.

---

## 📊 Experimental Dataset Allocation

The dataset contains **5,308 color images (RGB)** structured into three independent root directories to ensure absolute isolation between parameter optimization and final testing.

| Dataset Partition | Total Image Files | Target Structural Function |
| :--- | :--- | :--- |
| **Training Set** | 2,778 images | Updates network weights via gradient descent backpropagation loops. |
| **Validation Set** | 2,442 images | Monitors validation loss to adjust hyperparameter tuning and trigger early stopping. |
| **Test Set** | 88 images | Evaluates final, completely unbiased generalization accuracy on independent samples. |

All input shapes are dynamically normalized to a fixed tensor resolution of $128 \times 128 \times 3$ pixels using a linear scaling transformation:
$$x_{\text{normalized}} = \frac{x}{255.0}$$

---

## 🧱 Model Architecture Specifications

### 1. Custom VGG-Style CNN (From Scratch)
* **Feature Extraction:** 3 progressive convolutional blocks (32, 64, and 128 filters) utilizing $3 \times 3$ spatial kernels, ReLU activations, and Batch Normalization.
* **Downsampling:** Max pooling layers with a $2 \times 2$ pool size and a stride of 2.
* **Regularization Barrier:** Tiered Dropout layers ranging from 20% to 40% within the convolutional blocks, capped with a strong 50% Dropout layer on the dense classification head.

### 2. Transfer Learning Model (MobileNetV2)
* **Backbone Extractor:** MobileNetV2 loaded with pre-trained weights from the global **ImageNet** database (`include_top=False`).
* **Freezing Strategy:** The entire base convolutional structure is frozen (`base_model.trainable = False`), locking **2,257,920 parameters** to protect historical feature configurations.
* **Custom Head:** Integrated with a `GlobalAveragePooling2D` layer, structural Batch Normalization layers, sequential Dropout barriers (30% and 40%), a 128-unit hidden dense block, and a 6-unit Softmax output layer.

---

## 📈 Empirical Performance Matrix

| Model Performance Metric | Custom VGG-Style CNN | Pre-trained MobileNetV2 |
| :--- | :--- | :--- |
| **Total System Parameters** | 4,289,862 | 2,422,982 |
| **Trainable System Parameters** | 4,289,158 | **165,062** |
| **Non-Trainable Base Parameters** | 704 | 2,257,920 |
| **Best Logged Validation Accuracy** | 78.05% | **97.17%** |
| **Final Unbiased Test Accuracy** | 46.59% | **59.09%** |

### 🔍 Key Core Observations
1. **Transfer Learning Superiority:** MobileNetV2 achieved a peak validation accuracy of **97.17%**, outperforming the custom model while requiring only **165,062 trainable parameters**.
2. **The Generalization Gap:** Both architectures suffered a noticeable drop in accuracy when exposed to the 88-image test set, revealing significant data variance and domain shift between data partitions.

---

## 🖼️ Research Figures and Visualization

### Figure 1: Model Training History Curves
Plots tracking epoch-by-epoch learning trends. The left panel shows the classic validation accuracy plateau and overfitting gap of the custom CNN, while the right panel displays the tight, stable convergence of the pre-trained MobileNetV2 architecture.

http://googleusercontent.com/image_generation_content/0

### Figure 2: Evaluation Test Set Confusion Matrix
A 6-axis color-mapped confusion matrix evaluating MobileNetV2 performance on the isolated test set. The diagonal entries trace true positive distributions, while the off-diagonal entries isolate exact failure modes caused by domain shift.

http://googleusercontent.com/image_generation_content/1

### Figure 3: Deployed Streamlit Web Application Interface
The interactive, web-based production dashboard. The interface includes real-world tensor augmentation maps (horizontal/vertical flips, rotations, and zooms), live model architecture switching dropdowns, and dynamic Softmax classification confidence bar charts.

http://googleusercontent.com/image_generation_content/2

---

## 🚀 How to Run the Live App Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/fruit-degradation-cnn.git](https://github.com/yourusername/fruit-degradation-cnn.git)
cd fruit-degradation-cnn
