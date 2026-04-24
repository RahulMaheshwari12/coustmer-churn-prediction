# Customer Churn Prediction (Machine Learning Project)

## 📌 Overview

This project focuses on predicting customer churn for a telecom dataset using machine learning. The notebook performs complete data processing, exploratory data analysis (EDA), feature engineering, and model training using Logistic Regression.

---

## ⚙️ Workflow

### 1. Data Loading

* Training and test datasets are loaded from CSV files
* Originally designed to run on Google Colab using Google Drive

### 2. Data Preprocessing

* Data type optimization (float64 → float32, int64 → int32)
* Missing values removed using `dropna()`
* Duplicate rows removed
* Irrelevant columns dropped (`id`, `gender`)

---

### 3. Exploratory Data Analysis (EDA)

* Target distribution analysis (Churn)
* Categorical feature analysis
* Numerical feature distributions (histograms + KDE)
* Outlier detection using boxplots
* Correlation heatmap

---

### 4. Feature Engineering

* Target variable encoding:

  * Yes → 1
  * No → 0
* Categorical encoding:

  * Binary → Ordinal Encoding
  * Multi-category → One-Hot Encoding

---

### 5. Data Splitting

* Train-test split (80:20)

---

### 6. Feature Scaling

* StandardScaler for normal distributions
* PowerTransformer / MinMaxScaler for skewed features
* Applied using ColumnTransformer

---

### 7. Model Training

* Logistic Regression with `class_weight='balanced'`
* Custom training function used for:

  * Prediction
  * Evaluation

---

### 8. Hyperparameter Tuning

* RandomizedSearchCV with Stratified K-Fold
* Parameters tuned:

  * Regularization strength (C)
  * Solver type
  * Penalty (L1, L2)

---

### 9. Threshold Optimization

* Tested thresholds: 0.5, 0.6, 0.7, 0.8
* Final threshold selected: **0.6**

---

### 10. Final Prediction

* Predictions generated on test dataset
* Output converted to binary classification

---

### 11. Model Saving

* Final model saved as:

```
best_lr_model.pkl
```

---

## 📊 Output

* Classification report
* Churn predictions
* Saved trained model

---

## 🚫 Limitations (Important)

* Uses Google Colab-specific paths (not portable)
* Only Logistic Regression used (no model comparison)
* Missing value handling is simplistic (drop instead of imputation)
* No pipeline used (risk of data leakage)
* Code not modularized into reusable scripts

---

## ▶️ How to Run

### Step 1: Install dependencies

```
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Step 2: Update dataset paths

Replace:

```
/content/drive/MyDrive/rahul_data/
```

with your local dataset path.

### Step 3: Run notebook

```
jupyter notebook main.ipynb
```

---

## 📁 Project Structure

```
project/
│
├── main.ipynb
├── train.csv
├── test.csv
└── best_lr_model.pkl
```

---

## 🔮 Future Improvements

* Use Pipeline (sklearn)
* Add multiple models (Random Forest, XGBoost)
* Improve missing value handling
* Add ROC-AUC and cross-validation
* Convert notebook to production-ready code

---

## 👨‍💻 Author

Rahul Maheshwari
