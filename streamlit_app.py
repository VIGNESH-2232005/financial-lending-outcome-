import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('encoders.pkl', 'rb') as f:
            encoders = pickle.load(f)
        return model, scaler, encoders
    except Exception as e:
        st.error(f"Error loading artifacts: {e}")
        return None, None, None

model, scaler, encoders = load_artifacts()

# Helper to encode safely
def encode_val(col, val):
    if col in encoders:
        le = encoders[col]
        try:
            return le.transform([str(val)])[0]
        except:
            return 0 
    return val

st.title("FinLend AI - Loan Eligibility Predictor")
st.write("Enter applicant details to get an instant AI-powered credit decision.")

# Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Marital Status", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["No", "Yes"])
        
    with col2:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=50000)
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0)
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=200000)
        loan_term = st.number_input("Loan Amount Term (Months)", min_value=0, value=360)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "1.0 (Good)" if x == 1.0 else "0.0 (Bad)")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submitted = st.form_submit_button("Predict Eligibility")

if submitted:
    if model and scaler and encoders:
        # Prepare features
        features = []
        features.append(encode_val('applicant_gender', gender))
        features.append(encode_val('is_married', married))
        features.append(encode_val('num_dependents', dependents))
        features.append(encode_val('education_level', education))
        features.append(encode_val('is_self_employed', self_employed))
        
        features.append(float(applicant_income))
        features.append(float(coapplicant_income))
        
        # Model expects Loan Amount in 'Thousands'. 
        # User enters full amount (e.g. 200000). We divide by 1000.
        features.append(float(loan_amount) / 1000)
        
        features.append(float(loan_term))
        features.append(float(credit_history))
        features.append(encode_val('residence_area', property_area))
        
        final_features = [np.array(features)]
        
        if scaler:
            final_features = scaler.transform(final_features)
            
        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)[0]
        
        # Class 1 is usually "Approved" (Y)
        prob_approved = probability[1]
        prob_rejected = probability[0]
        
        result = "Approved" if prediction[0] == 1 else "Rejected"
        
        if 'approval_status' in encoders:
            result_val = encoders['approval_status'].inverse_transform(prediction)[0]
            if result_val == 'Y': result = "Approved"
            elif result_val == 'N': result = "Rejected"
            
        st.subheader("Prediction Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Probability of Approval", f"{prob_approved:.1%}")
        with col2:
            st.metric("Probability of Rejection", f"{prob_rejected:.1%}")
            
        st.write("---")
        
        if result == "Approved":
            st.success(f"Result: **APPROVED**")
            st.progress(prob_approved)
        else:
            st.error(f"Result: **REJECTED**")
            st.progress(prob_approved)
            st.info("Tip: To improve your chances, try increasing income or opting for a longer loan term.")
            
        # --- Rule-Based Analysis ---
        st.write("---")
        st.subheader("Why this decision? (Rule Check)")
        
        # 1. Credit History Rule
        if credit_history == 1.0:
            st.success("✅ **Credit History:** PASS (Good Payment History)")
        else:
            st.error("❌ **Credit History:** FAIL (Bad History is a Major Rejection Factor)")
            
        # 2. Affordability Rule (EMI Calculation)
        # Assuming 8.5% Annual Interest Rate for estimation
        r = 0.085 / 12
        n = loan_term
        P = float(loan_amount) # This is in Rupees
        
        if P > 0 and n > 0:
            # EMI Formula
            emi = P * r * (pow(1 + r, n)) / (pow(1 + r, n) - 1)
            total_income = float(applicant_income) + float(coapplicant_income)
            
            # Affordability Threshold: EMI should not exceed 50% of Income
            if total_income > 0:
                emi_ratio = emi / total_income
                
                col_emi1, col_emi2 = st.columns(2)
                with col_emi1:
                    st.write(f"**Estimated EMI:** ₹{emi:,.0f}/month")
                with col_emi2:
                    st.write(f"**Total Income:** ₹{total_income:,.0f}/month")
                
                if emi_ratio > 0.60: # Using 60% as a slightly lenient cut-off
                    st.error(f"❌ **Affordability:** FAIL (EMI is {emi_ratio:.0%} of Income, recommended < 50%)")
                elif emi_ratio > 0.50:
                    st.warning(f"⚠️ **Affordability:** RISKY (EMI is {emi_ratio:.0%} of Income)")
                else:
                    st.success(f"✅ **Affordability:** PASS (EMI is {emi_ratio:.0%} of Income)")
            else:
                st.error("❌ **Income:** FAIL (Total Income is 0)")
        else:
             st.warning("Cannot calculate EMI (Loan or Term is 0)")

    else:
        st.error("Model artifacts not loaded.")
