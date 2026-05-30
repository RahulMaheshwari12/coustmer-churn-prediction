# AI Customer Churn Prediction & Attrition Dashboard

An end-to-end Machine Learning pipeline and interactive Flask web dashboard for analyzing and predicting customer attrition. This project is built using Python, Scikit-Learn, LightGBM, and Flask, featuring customized preprocessing layers and a premium dark-themed web interface.

---

## 🌟 Key Features

* **Advanced Feature Preprocessing**: 
  * **Encoding**: Categorical binary columns encoded via `OrdinalEncoder`, multi-category columns encoded via `OneHotEncoder(handle_unknown='ignore')`.
  * **Custom Scaling**: Applies different scalers based on feature distributions (`StandardScaler` for normal, `PowerTransformer/Yeo-Johnson` for skewed charges, and `MinMaxScaler` for bimodal tenure values).
* **Robust Modeling**: Baseline benchmarking across Logistic Regression, Random Forest, XGBoost, LightGBM, and CatBoost with imbalanced-class weight scaling.
* **Interactive Flask Dashboard**:
  * **Single Prediction Profile**: Dynamically updates text inputs, estimates Total Charges, and displays Churn Risk on an animated, color-coded SVG circular gauge alongside risk explanation points.
  * **Batch CSV Uploads**: Supports drag-and-drop CSV uploads, runs inference in batch, computes portfolio metrics, displays a **Chart.js Donut Chart**, and generates a downloadable output CSV.

---

## 📂 Project Structure

```text
├── app.py                             # Flask web app backend server
├── main.ipynb                         # Upgraded machine learning pipeline notebook
├── churn_preprocessor_encoding.joblib # Fitted categorical transformer pipeline
├── churn_preprocessor                 # Fitted numerical scaling pipeline
├── best_lightgbm_model.pkl            # Tuned LightGBM model binary
├── best_logistic_model.pkl            # Tuned Logistic Regression model binary
├── templates/
│   └── index.html                     # Responsive glassmorphic layout
├── static/
│   ├── css/style.css                  # Custom Vanilla CSS styling
│   └── js/main.js                     # UI animations, SVG gauges, and Chart.js integration
└── .gitignore                         # Config to exclude large datasets from git
```

---

## 📊 Model Attrition Results

Target class imbalance (~22.5% Churn / 77.5% Retained) was adjusted during training using positive-class scaling weights ($w \approx 3.417$):

| Model Name | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :---: | :---: | :---: | :---: |
| **LightGBM Classifier** | 82% | 0.57 | **0.88** | **0.69** |
| **XGBoost Classifier** | 82% | 0.57 | 0.87 | **0.69** |
| **Logistic Regression** | 81% | 0.55 | **0.88** | 0.68 |
| **Random Forest Classifier** | **84%** | **0.67** | 0.58 | 0.63 |

*Note: LightGBM was selected as the production model due to its optimal balance between Recall and Precision.*

---

## 🚀 Quick Start Guide

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install flask joblib pandas numpy scikit-learn lightgbm xgboost catboost matplotlib seaborn
```

### 2. Run the Dashboard locally
Initialize the Flask server:
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

### 3. Running Batch Predictions
Upload a CSV dataset (columns matching `test.csv` features) via the dashboard to run batch inference. Download the output file `submission.csv` containing predicted labels and churn probabilities.
