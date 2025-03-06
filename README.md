# 🎯 Breast_Cancer_Prediction

This repository presents a **machine learning approach** for predicting:  
- **Relapse-Free Survival (RFS)** (regression)  
- **Pathological Complete Response (PCR)** (classification)  

The dataset is derived from the **I-SPY2 TRIAL**, containing **clinical and imaging-based features** for **400 breast cancer patients**. This study aims to assist in **patient stratification** and **treatment decision-making** by leveraging **ML models** to predict relapse and treatment response.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)  
2. [Repository Structure](#-repository-structure)  
3. [Data Description](#-data-description)  
4. [Methodology](#-methodology)  
5. [Usage Instructions](#-usage-instructions)  
6. [Requirements](#-requirements)  
7. [Results](#-results)  
8. [Contributing](#-contributing)  
9. [License](#-license)  
10. [Author](#-author)  

---

## 🔍 Project Overview

- **Relapse-Free Survival (RFS)**: A continuous variable representing the time before a patient relapses post-treatment.  
- **Pathological Complete Response (PCR)**: A **binary** outcome indicating whether a tumour **completely disappears** after neoadjuvant therapy.  
- **Challenge**: The dataset is **imbalanced** (~22% positive PCR cases).  

### 🎯 Key Objectives
1. **Regression**: Predict RFS using models like **Random Forest**, **SVM**, and **XGBoost**.  
2. **Classification**: Predict PCR using **cost-sensitive** approaches (Weighted Random Forest, XGBoost, Logistic Regression) to handle class imbalance.  
3. **Evaluation Metrics**:  
   - **Regression** → Mean Absolute Error (MAE)  
   - **Classification** → Balanced Accuracy, F1-score  

---

## 📂 Repository Structure
```
BreastCancer_Prediction/
├── notebooks/
│   ├── Reg_Train_RFS.ipynb  # Notebook for regression models (RFS)
│   ├── CLF_Train_PCR.ipynb  # Notebook for classification models (PCR)
├── Predicting_RFS_PCR_for_Breast_Cancer_Patients.pdf
└── README.md
```

---

## 📊 Data Description

- **Dataset**: I-SPY2 TRIAL dataset with **400 samples**.  
- **Feature Categories**:  
  - **Clinical**: Age, tumour stage, hormone receptor status, etc.  
  - **Imaging-based**: MRI-derived features (**108+ continuous variables**).  
- **Outcome Variables**:  
  - **RFS (Continuous)**  
  - **PCR (Binary, Imbalanced: 22% PCR-positive cases)**  

---

## ⚙️ Methodology

### **Regression (RFS)**
- **Random Forest** → Optimized `n_estimators`, `max_features`.  
- **SVM** → PCA for dimensionality reduction, tuned `C`, `epsilon`, and `kernel`.  
- **XGBoost** → Grid search for `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`.  

### **Classification (PCR)**
- **Logistic Regression** → Cost-sensitive learning, L2 regularization.  
- **Random Forest** → Weighted classes for better minority-class recall.  
- **XGBoost** → `scale_pos_weight` to correct for class imbalance.  
- **Voting Classifier** → Combined LR, RF, XGB with **soft voting**.  

---

## 🚀 Usage Instructions

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/YOUR_USERNAME/BreastCancer_Prediction.git
cd BreastCancer_Prediction
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3️⃣ Run Jupyter Notebooks**
```bash
jupyter notebook
```
- Open `notebooks/Reg_Train_RFS.ipynb` (for RFS regression).  
- Open `notebooks/CLF_Train_PCR.ipynb` (for PCR classification).  

### **4️⃣ Explore Results**
- The notebooks contain visualizations & evaluation metrics for each model.  
- Refer to the PDF report for a detailed breakdown.  

---

## 📦 Requirements

Below is a typical `requirements.txt`:

```txt
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
imbalanced-learn
jupyter
```

---

## 📈 Results

### **1️⃣ Regression (RFS)**
- **XGBoost** achieved the lowest MAE with stable cross-validation.  
- **SVM** and **Random Forest** performed similarly but had higher variance.  

### **2️⃣ Classification (PCR)**
- **Voting Ensemble (LR, RF, XGB)** had the highest balanced accuracy (~61-62%).  
- **Class imbalance significantly impacted minority-class recall.**  

| Model            | Metric             | Score (Test) |
|-----------------|------------------|--------------|
| **XGBoost (RFS)** | MAE (Regression)  | ~25          |
| **Voting Ensemble** | Balanced Accuracy | ~0.61        |

---

## 🤝 Contributing

1. **Fork** this repo.  
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new model or fix bug"
   ```
4. Push to GitHub:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a **Pull Request**.


