# 🧮 STAT 4540: Machine Learning & Statistical Analysis

## 📌 Overview
This project applies **machine learning, dimensionality reduction, clustering, and regression techniques** to solve real-world classification and prediction problems. The analysis covers:
- **MNIST Classification**: Using K-NN, LDA, QDA, and Naive Bayes.
- **Principal Component Analysis (PCA)**: Dimensionality reduction for image recognition.
- **Clustering Methods**: Hierarchical clustering & 2-Means clustering.
- **High-Dimensional Regression**: Variable selection, ridge/lasso regression, and PCR/PLS.

---

## 📥 Large File Handling
The MNIST training dataset (`train_mnist.csv`) is over 100MB and exceeds GitHub's file size limit. This file is excluded from the repository. To work with this project:

1. **Download the file separately**: The file can be downloaded from [Kaggle MNIST Dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or other MNIST sources.
2. **Place the file in the correct location**: After downloading, place `train_mnist.csv` in the `data/` directory.

Alternatively, you can generate the MNIST dataset using Python:
```python
from sklearn.datasets import fetch_openml
import pandas as pd

# Download MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data
y = mnist.target

# Create training dataset (first 10,000 samples)
train_data = pd.concat([y.iloc[:10000], X.iloc[:10000]], axis=1)
train_data.to_csv('data/train_mnist.csv', index=False)
```

---

---

## 📜 Dataset
- **MNIST Handwritten Digits Dataset**: [Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)
  - **Train**: `train_mnist.csv` (10,000 samples)
  - **Test**: `test_mnist.csv` (1,000 samples)
  - **Features**: 784 pixel values (vectorized 28x28 image)
  - **Target Variable**: Digit classification (0-9)
- **Regression Dataset**:
  - **Train**: `reg_train.csv` (200 samples)
  - **Test**: `reg_test.csv` (20 samples)
  - **Features**: 500 predictors (x1, x2, ..., x500)
  - **Target Variable**: `y`

---

## 🔍 MNIST Classification (Question 1)
### **Methods Used**
- **K-Nearest Neighbors (K-NN)**: Find optimal K and test error rate.
- **Linear Discriminant Analysis (LDA)**: Compute discriminant function for each digit.
- **Quadratic Discriminant Analysis (QDA)**: Higher flexibility for class separation.
- **Naive Bayes**: Uses `e1071` package for posterior probability estimation.

### **Key Results**
📌 **Best Classifier**: Selected based on **error rate & class-specific True Positive Rates (TPRs)**.

---

## 📊 PCA for MNIST (Question 2)
### **Objective**
- Reduce image dimensionality using **Principal Component Analysis (PCA)**.
- Approximate images using the top **M** principal components.

### **Steps**
1. Compute **PCA** on digits **1 and 8** separately.
2. Analyze variance explained by **ϕ1, ϕ2 (Principal Components)**.
3. **Reconstruct images** using **M = 1, 2, ..., 500** PCs.
4. **Plot reconstruction error** for different M values.

### **Findings**
✅ PCA successfully compresses MNIST data while retaining digit structures.  
✅ Digits **8 & 1** show different PCA patterns due to geometric differences.

---

## 🏷️ Clustering (Question 3)
### **Techniques Used**
- **Hierarchical Clustering**: Using **complete, average, and single linkage**.
- **2-Means Clustering**: Applied on **raw images & PCA-transformed data**.

### **Key Results**
📌 Clustering on **PCA-transformed data** improves digit separation compared to raw images.  
📌 **Complete-linkage clustering** provides more balanced dendrograms.

---

## 📈 High-Dimensional Regression (Question 4)
### **Variable Selection Methods**
- **Forward Selection**: Selects best predictors iteratively.
- **Backward Selection**: Eliminates least significant predictors.

### **Regularization Methods**
- **Ridge Regression**: Uses L2 penalty to reduce overfitting.
- **Lasso Regression**: Uses L1 penalty for feature selection.

### **Key Results**
📌 **Lasso Regression** selects **fewer, meaningful predictors**, improving interpretability.  
📌 **PCR & PLS Regression** compare dimensionality reduction for regression tasks.  
📌 Best model selected based on **Bias-Variance tradeoff** & **Test MSE**.

---
