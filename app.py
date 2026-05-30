import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load preprocessors and model
try:
    pre_encoding = joblib.load('churn_preprocessor_encoding.joblib')
    pre_scaler = joblib.load('churn_preprocessor')
    model = joblib.load('best_lightgbm_model.pkl')
    print("Model and preprocessors loaded successfully!")
except Exception as e:
    print(f"Error loading models: {e}")
    pre_encoding, pre_scaler, model = None, None, None

# Reconstruct column lists matching train set exactly
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
              'Contract', 'PaymentMethod']
other_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

# Calculate final encoded column order
oh_feature_names = list(pre_encoding.named_transformers_['multi'].get_feature_names_out(multi_cols))
all_final_cols = binary_cols + oh_feature_names + other_cols

# Calculate final scaled column order
transformed_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
remainder_cols = [col for col in all_final_cols if col not in transformed_cols]
all_scale_columns = transformed_cols + remainder_cols

def preprocess_data(input_df):
    # 1. Encoding
    encoded_data = pre_encoding.transform(input_df)
    encoded_df = pd.DataFrame(encoded_data, columns=all_final_cols, index=input_df.index)
    
    # 2. Scaling
    scaled_data = pre_scaler.transform(encoded_df)
    scaled_df = pd.DataFrame(scaled_data, columns=all_scale_columns, index=input_df.index)
    
    # 3. Align with model training columns order
    scaled_df = scaled_df[all_scale_columns]
    return scaled_df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_single', methods=['POST'])
def predict_single():
    if not model:
        return jsonify({'error': 'Model not loaded on server'}), 500
    
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        
        # preprocess
        processed_df = preprocess_data(input_df)
        
        # predict
        pred = model.predict(processed_df)[0]
        prob = model.predict_proba(processed_df)[0][1]
        
        # Explain churn reasons
        reasons = []
        if float(data['MonthlyCharges']) > 80:
            reasons.append("High Monthly Charges (> $80) increase overall churn likelihood.")
        if int(data['tenure']) <= 12:
            reasons.append("Customer is within critical first year of service (tenure <= 12).")
        if data['Contract'] == 'Month-to-month':
            reasons.append("Flexible Month-to-month contracts exhibit high historical churning rates.")
        if data['TechSupport'] == 'No':
            reasons.append("No technical support subscription increases risk of customer friction.")
        if data['InternetService'] == 'Fiber optic' and float(data['MonthlyCharges']) > 90:
            reasons.append("Fiber optic subscription with high billing is associated with high churn rates.")

        if not reasons:
            reasons.append("Customer displays stable retention indicators.")

        return jsonify({
            'churn_prediction': int(pred),
            'churn_probability': float(prob),
            'risk_level': 'High' if prob > 0.60 else ('Medium' if prob > 0.35 else 'Low'),
            'reasons': reasons
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
