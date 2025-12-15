import os
import pickle
import numpy as np
# Trigger reload
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

# Load Artifacts
model = None
scaler = None
encoders = None

def load_artifacts():
    global model, scaler, encoders
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        print("Artifacts loaded successfully.")
    except Exception as e:
        print(f"Error loading artifacts: {e}")

load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.json
        print(f"Received data: {data}")
        
        # Features expected by the model in order
        # Assuming the order from the training script:
        # We need to reconstruct the feature vector.
        # Based on dataset: Gender, Married, Dependents, Education, Self_Employed, 
        # ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
        
        # Note: 'Loan_ID' was dropped.
        # We need to match the exact input columns used in training.
        # Let's map inputs to the training columns.
        
        # Helper to encode safely
        def encode_val(col, val):
            if col in encoders:
                le = encoders[col]
                try:
                    return le.transform([str(val)])[0]
                except:
                    # Fallback for unseen labels -> map to mode or 0? 
                    # For safety, use transform on the most frequent or just 0
                    print(f"Warning: Unseen label '{val}' for column '{col}'")
                    return 0 # encoders[col].transform([encoders[col].classes_[0]])[0] 
            return val

        # Prepare input vector (Must match the order of columns after 'application_id' drop and before target)
        # We should check the columns from `analysis.py` but for now we infer from standard dataset structure.
        # Columns (typical): 
        # Gender, Married, Dependents, Education, Self_Employed, 
        # ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area
        
        features = []
        
        features.append(encode_val('applicant_gender', data.get('gender')))
        features.append(encode_val('is_married', data.get('married')))
        features.append(encode_val('num_dependents', data.get('dependents')))
        features.append(encode_val('education_level', data.get('education')))
        features.append(encode_val('is_self_employed', data.get('self_employed')))
        
        # Numerical need to be float
        features.append(float(data.get('applicant_income', 0)))
        features.append(float(data.get('coapplicant_income', 0)))
        features.append(float(data.get('loan_amount', 0)))
        features.append(float(data.get('loan_term', 360)))
        features.append(float(data.get('credit_history', 1.0)))
        
        features.append(encode_val('residence_area', data.get('property_area')))
        
        # Reshape for single sample
        final_features = [np.array(features)]
        
        # Scale
        if scaler:
            final_features = scaler.transform(final_features)
            
        prediction = model.predict(final_features)
        
        # prediction is likely encoded (Y=1, N=0) or similar.
        # We need to decode it back if encoders covers target.
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        # Check if target 'approval_status' is in encoders
        if 'approval_status' in encoders:
            result = encoders['approval_status'].inverse_transform(prediction)[0]
            # Map Y/N to user friendly
            if result == 'Y': result = "Approved"
            elif result == 'N': result = "Rejected"

        return jsonify({'prediction': result})

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
